"""MLP-style conceptual animation of screening and masked teacher repair.

No datasets, fitted checkpoints, measured scores or training are used. The
weights and screening outcomes below are deliberately illustrative. A single
MLP unit's incoming weights, hidden bias and outgoing weights form one toy
editable group. The real experiment uses ResNet channel-coordinate unions.
"""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


N_INPUT = 3
N_HIDDEN = 10
N_OUTPUT = 2
TOTAL_PARAMETERS = 62  # 3*10 incoming + 10 hidden biases + 10*2 outgoing + 2 output biases.
STRICT_RANKING = (6, 2)  # Displayed as H7, H3: passed the hypothetical ablations.
FALLBACK_RANKING = (4, 8, 0, 1, 3, 5, 7, 9)
STABLE_UNITS = (2, 4, 6)
TEST_ORDER = (6, 4, 2)
TEST_OUTCOMES = {6: 'P', 4: 'F', 2: 'P'}
BOOTSTRAP_VOTES = (3, 2, 17, 2, 16, 2, 15, 1, 1, 1)  # 60 = ceil(25%*10)*20 votes.
COMPACT_DURATIONS = (2.5, 3.0, 2.0, 2.0, 2.0, 4.0, 5.0, 5.0, 4.0)
DURATIONS = (6.0, 10.0, 7.0, 7.0, 7.0, 12.0, 20.0, 12.0, 9.0)
TOTAL_SECONDS = sum(DURATIONS)
PHASE_TITLES = (
    '1  Start with an existing teacher',
    '2  Find stable candidate units',
    '3  Screen H7: temporarily switch its output off',
    '3  Screen H5: temporarily switch its output off',
    '3  Screen H3: temporarily switch its output off',
    '4  Choose which existing weights may change',
    '5  Repair: update only the selected weight groups',
    '6  Apply the same validation checks to both arms',
    '7  If accepted, freeze the teacher and run ordinary KD',
)
PALETTE = {'bg': '#f7f9fc', 'ink': '#172b46', 'muted': '#57697e',
           'line': '#9cabbc', 'node': '#e8eef5', 'pass': '#16866b',
           'fail': '#bc7a17', 'stable': '#7955ad', 'extra': '#2869c7',
           'off': '#c54745', 'white': '#ffffff'}


@dataclass(frozen=True)
class StoryState:
    phase: int
    progress: float
    off_unit: int | None
    outcomes: tuple[tuple[int, str], ...]
    cap: int
    strict_units: tuple[int, ...]
    expanded_units: tuple[int, ...]
    repair_progress: float


def parameter_mask(units):
    """Actual union of the toy MLP's associated scalar weight/bias coordinates."""
    mask = np.zeros(TOTAL_PARAMETERS, dtype=bool)
    for unit in units:
        if isinstance(unit, bool) or not isinstance(unit, (int, np.integer)) or not 0 <= unit < N_HIDDEN:
            raise ValueError('Toy hidden-unit indices must be integers from 0 to 9')
        mask[unit * N_INPUT:(unit + 1) * N_INPUT] = True
        mask[30 + unit] = True
        mask[40 + unit] = True
        mask[50 + unit] = True
    return mask


def select_units(policy, cap):
    """Longest ranked prefix under an integer toy coordinate cap, not a real cap."""
    if policy not in {'strict', 'expanded'}:
        raise ValueError('Unknown toy policy')
    ranking = STRICT_RANKING + (FALLBACK_RANKING if policy == 'expanded' else ())
    selected = ()
    for unit in ranking:
        proposal = selected + (unit,)
        if int(parameter_mask(proposal).sum()) > cap:
            break
        selected = proposal
    return selected


def base_weights():
    return np.random.default_rng(2026).uniform(.22, .70, TOTAL_PARAMETERS)


def repaired_weights(units, progress):
    """Illustrative interpolation ONLY; this is not an SGD or KD simulation."""
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError('Repair progress must be between zero and one')
    values = base_weights()
    direction = .10 * np.sin(np.arange(TOTAL_PARAMETERS) * .91 + .7)
    mask = parameter_mask(units)
    values[mask] += progress * direction[mask]
    return values


def state_at(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Animation time must be finite and nonnegative')
    elapsed = 0.0
    phase = len(DURATIONS) - 1
    progress = 1.0
    for index, duration in enumerate(DURATIONS):
        if seconds < elapsed + duration:
            phase, progress = index, (seconds - elapsed) / duration
            break
        elapsed += duration
    outcomes = []
    off = None
    if 2 <= phase <= 4:
        test = phase - 2
        outcomes.extend((unit, TEST_OUTCOMES[unit]) for unit in TEST_ORDER[:test])
        if progress < .55:
            off = TEST_ORDER[test]
        else:
            outcomes.append((TEST_ORDER[test], TEST_OUTCOMES[TEST_ORDER[test]]))
    elif phase >= 5:
        outcomes = list(TEST_OUTCOMES.items())
    cap = (12 if progress < .35 else 18 if progress < .65 else 24) if phase == 5 else 24
    strict = select_units('strict', cap) if phase >= 5 else ()
    expanded = select_units('expanded', cap) if phase >= 5 else ()
    # Hold the final repaired weights before validation starts. This makes the
    # audit show zero weight updates during validation/KD, including boundaries.
    repair = min(1.0, progress / .9) if phase == 6 else 1.0 if phase > 6 else 0.0
    return StoryState(phase, progress, off, tuple(outcomes), cap, strict, expanded, repair)


def _text(ax, x, y, message, size=13, color=None, **kwargs):
    return ax.text(x, y, message, fontsize=size, color=color or PALETTE['ink'],
                   va='center', fontfamily='DejaVu Sans', **kwargs)


def draw_network(ax, left, width, state, units=(), *, low=13, high=40,
                 repair=0.0, names=True, radius=1.0, show_status=True):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    xs = [left + width * fraction for fraction in (.12, .50, .88)]
    ys = [np.linspace(low + .24 * (high - low), high - .24 * (high - low), N_INPUT),
          np.linspace(high, low, N_HIDDEN),
          np.linspace(low + .34 * (high - low), high - .34 * (high - low), N_OUTPUT)]
    weights = repaired_weights(units, repair)
    segments, colors, widths = [], [], []
    for unit in range(N_HIDDEN):
        selected = unit in units
        color = PALETTE['pass'] if unit in STRICT_RANKING else PALETTE['extra']
        for incoming in range(N_INPUT):
            index = N_INPUT * unit + incoming
            segments.append([(xs[0], ys[0][incoming]), (xs[1], ys[1][unit])])
            colors.append((color, .80) if selected else (PALETTE['line'], .23))
            widths.append(.65 + (3.4 if selected else 1.0) * abs(weights[index]))
        for outgoing in range(N_OUTPUT):
            index = 40 + N_HIDDEN * outgoing + unit
            segments.append([(xs[1], ys[1][unit]), (xs[2], ys[2][outgoing])])
            if unit == state.off_unit:
                colors.append((PALETTE['line'], .04))
                widths.append(.6)
            else:
                colors.append((color, .80) if selected else (PALETTE['line'], .23))
                widths.append(.65 + (3.4 if selected else 1.0) * abs(weights[index]))
    from matplotlib.colors import to_rgba
    ax.add_collection(LineCollection(segments, colors=[to_rgba(c, a) for c, a in colors],
                                     linewidths=widths, zorder=1))
    outcomes = dict(state.outcomes)
    for layer, positions in enumerate(ys):
        for index, y in enumerate(positions):
            face, edge = PALETTE['node'], PALETTE['line']
            linewidth = 1.0
            if layer == 1:
                if outcomes.get(index) == 'P':face = '#d5eee6'
                elif outcomes.get(index) == 'F':face = '#fae8cb'
                elif state.phase >= 1 and index in STABLE_UNITS:face = '#e8e0f3'
                if index in units:
                    edge = PALETTE['pass'] if index in STRICT_RANKING else PALETTE['extra']
                    linewidth = 2.5
                if index == state.off_unit:
                    face, edge, linewidth = PALETTE['white'], PALETTE['off'], 2.5
            ax.add_patch(Circle((xs[layer], y), radius, facecolor=face,
                               edgecolor=edge, linewidth=linewidth, zorder=3))
            if names:
                label = f'x{index+1}' if layer == 0 else f'H{index+1}' if layer == 1 else f'y{index+1}'
                if layer == 1 and index == state.off_unit:label = '0'
                _text(ax, xs[layer], y, label, 10.5, ha='center', zorder=4)
            if layer == 1 and show_status and state.phase >= 1:
                mark = ('OFF' if index == state.off_unit else outcomes.get(index,
                        'S' if index in STABLE_UNITS and state.phase < 5 else '?'))
                mark_color = (PALETTE['off'] if mark == 'OFF' else PALETTE['pass'] if mark == 'P'
                              else PALETTE['fail'] if mark == 'F' else PALETTE['muted'])
                _text(ax, xs[layer] + radius + .6, y, mark, 10, color=mark_color)
    return xs, ys


def support_bar(ax, left, width, units, cap):
    from matplotlib.patches import Rectangle
    # The illustrated MLP groups have exactly six coordinates each, unlike ResNet groups.
    for index in range(cap // 6):
        unit = units[index] if index < len(units) else None
        color = (PALETTE['pass'] if unit in STRICT_RANKING else PALETTE['extra']) if unit is not None else PALETTE['node']
        ax.add_patch(Rectangle((left + width * index / 4, 8.3), width / 4 - .7, 1.6,
                               facecolor=color, edgecolor='none'))
    count = int(parameter_mask(units).sum())
    _text(ax, left, 6.6, f'{count} editable / {cap} allowed coordinates; others frozen', 11)


def draw_frame(fig, ax, seconds):
    state = state_at(seconds)
    ax.clear();ax.set_xlim(0, 100);ax.set_ylim(0, 56.25);ax.set_aspect('equal');ax.axis('off')
    fig.patch.set_facecolor(PALETTE['bg']);ax.set_facecolor(PALETTE['bg'])
    _text(ax, 3, 53.7, PHASE_TITLES[state.phase], 21, fontweight='bold')
    subtitle = ''
    detail = ''
    if state.phase <= 4:
        xs, ys = draw_network(ax, 19, 62, state)
        for x, label in zip(xs, ('Inputs', '10 existing hidden units', 'Outputs')):
            _text(ax, x, 44, label, 12, ha='center')
        if state.phase == 0:
            subtitle = 'MLP analogy: connections are weights. The teacher is already available.'
            detail = 'No new layers or neurons are added during repair.'
            # A schematic forward signal, not computed model activations.
            x = xs[0] + (xs[2] - xs[0]) * state.progress
            ax.scatter([x], [ys[0][1]], s=80, color=PALETTE['extra'], zorder=5)
        elif state.phase == 1:
            subtitle = 'Repeat ranking 20 times on resampled examples; look for units consistently in the top 25%.'
            detail = 'S = stable: top-ranked at least 14/20 times. Other units have not been ablation-tested.'
            _text(ax, 50, 8.7, 'H3: 17/20 votes     H5: 16/20     H7: 15/20', 13, ha='center')
        else:
            unit = TEST_ORDER[state.phase - 2]
            subtitle = 'Ablation means setting one unit output to ZERO, temporarily. No weights change.'
            if state.off_unit is not None:
                detail = f'H{unit+1} is OFF. Compare target scores and preservation accuracy.'
            else:
                result = TEST_OUTCOMES[unit]
                detail = (f'H{unit+1} restored: PASS. Target margin improved; preservation guard passed.'
                          if result == 'P' else
                          f'H{unit+1} restored: FAIL. Its removal did not satisfy the screening rule.')
            _text(ax, 50, 8.7, 'P = passed ablation     F = failed ablation     ? = not tested', 12, ha='center')
    elif state.phase in (5, 6):
        subtitle = (f'Same toy allowance: {state.cap} of 62 parameter coordinates in either arm.'
                    if state.phase == 5 else
                    'Colored connections and their hidden biases may change. Gray weights stay fixed.')
        for left, title, units in ((2, 'STRICT', state.strict_units), (52, 'EXPANDED', state.expanded_units)):
            _text(ax, left + 23, 45.5, title, 17, ha='center', fontweight='bold')
            draw_network(ax, left, 46, state, units, repair=state.repair_progress)
            support_bar(ax, left + 2, 41, units, state.cap)
            if state.phase == 6:
                index = STRICT_RANKING[0] * N_INPUT
                before, after = base_weights()[index], repaired_weights(units, state.repair_progress)[index]
                _text(ax, left + 23, 42.8, f'Example selected weight: {before:.3f} → {after:.3f}', 11, ha='center')
        _text(ax, 50, 11, 'P = passed ablation     F = failed ablation     ? = not tested', 10.3, ha='center')
        detail = ('Strict stops at H7 + H3. Expanded can then add ranked H5 and H9, despite no passing ablation.'
                  if state.phase == 5 else
                  'Illustrative weight motion, not an optimizer simulation. Every hidden unit remains present.')
    elif state.phase == 7:
        subtitle = 'More editable weights are not permission to damage the teacher.'
        for left, title, units in ((2, 'STRICT', state.strict_units), (52, 'EXPANDED', state.expanded_units)):
            _text(ax, left + 23, 45.5, title, 17, ha='center', fontweight='bold')
            draw_network(ax, left, 46, state, units, low=25, high=40,
                         repair=1, radius=.65, names=False, show_status=False)
            _text(ax, left + 23, 22.6, f'{int(parameter_mask(units).sum())} edited-coordinate support', 11, ha='center')
        checks = ('1  Target recall (recognizing target classes) must improve',
                  '2  Overall accuracy loss ≤ 0.5 percentage points',
                  '3  Predicted probabilities must stay close: mean KL ≤ 0.01')
        for index, message in enumerate(checks):
            _text(ax, 5, 17.8 - 3.8 * index, message, 12.5)
        _text(ax, 66, 16, 'PASS → freeze repaired teacher', 12, color=PALETTE['pass'])
        _text(ax, 66, 11, 'FAIL → keep original teacher', 12, color=PALETTE['off'])
        detail = 'Identical final checks for both arms. Pass/fail outcomes are not computed in this animation.'
    else:
        subtitle = 'The smaller student and its KD recipe are identical in both arms.'
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle
        for center, title, units in ((36, 'STRICT arm — if accepted', state.strict_units),
                                     (15, 'EXPANDED arm — if accepted', state.expanded_units)):
            _text(ax, 3, center + 10, title, 13, fontweight='bold')
            draw_network(ax, 2, 31, state, units, low=center-6.7, high=center+6.7,
                         repair=1, radius=.48, names=False, show_status=False)
            _text(ax, 18, center-8.3, 'Frozen teacher (10 existing units)', 10.5, ha='center')
            ax.annotate('', xy=(65, center), xytext=(37, center),
                        arrowprops={'arrowstyle': '->', 'color': PALETTE['extra'], 'lw': 2})
            _text(ax, 51, center+2.3, 'Same KD training', 12, ha='center')
            ax.scatter([38 + 25 * state.progress], [center], s=55, color=PALETTE['extra'], zorder=5)
            points = [[(70, y) for y in np.linspace(center-4, center+4, 3)],
                      [(81, y) for y in np.linspace(center-5, center+5, 4)],
                      [(92, y) for y in (center-2, center+2)]]
            edges = [[a, b] for first, second in zip(points[:-1], points[1:]) for a in first for b in second]
            ax.add_collection(LineCollection(edges, colors=PALETTE['line'], linewidths=.7, alpha=.5))
            for layer in points:
                for x, y in layer:ax.add_patch(Circle((x, y), .7, facecolor='#dce8fa', edgecolor=PALETTE['extra']))
            _text(ax, 81, center-8.3, 'Same smaller student', 11, ha='center')
        detail = 'If repair is rejected: retain the original teacher/KD baseline; do not fit new repaired-teacher students.'
    _text(ax, 3, 50.1, subtitle, 12.5)
    _text(ax, 50, 3.9, detail, 11.7, ha='center')
    _text(ax, 50, 1.25,
          'Conceptual toy, not measured results. Real ResNet channels use mask-union caps: 0.4%, 0.8%, 1.6%, 3.2%.',
          10.3, color=PALETTE['muted'], ha='center')
    return state


def make_figure(dpi=80):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax


STEP_NOTES = (
    ('The teacher already exists; no refitting here.',
     'H1–H10 are existing candidate hidden units.',
     'P001–P062 identify every weight and bias.'),
    ('Repeat rankings on resampled examples.',
     'A ranking vote is NOT a parameter update.',
     'Only stable units receive an ablation test.'),
    ('Temporarily force one unit output to zero.',
     'Then restore its original output operation.',
     'Its weights stay unchanged throughout.'),
    ('A failed removal test is not a failed repair.',
     'Turning a unit off differs from updating it.',
     'We record failure; we do not erase the unit.'),
    ('Both passing units join the strict ranking.',
     'Untested units are NOT classified as failed.',
     'These screening outcomes are illustrative.'),
    ('Each toy unit owns six scalar coordinates.',
     'Both arms have the same allowed budget.',
     'Strict may run out of eligible support.'),
    ('Only coordinates inside each mask can move.',
     'Counts and delta statistics are computed.',
     'Weight motion is illustration, NOT SGD.'),
    ('Acceptance uses validation, not test data.',
     'Toy accuracy/recall/KL are NOT evaluated.',
     'Neither policy is guaranteed to pass.'),
    ('The accepted teacher becomes frozen again.',
     'The smaller student and KD recipe stay fixed.',
     'A rejected repair needs no new KD student.'),
)


def make_detailed_figure(dpi=80):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(24, 13.5), dpi=dpi)
    return fig, (fig.add_axes([0, .345, .64, .64]),
                 fig.add_axes([.66, .345, .33, .64]),
                 fig.add_axes([.02, .025, .96, .29]))


def _table(ax, rows, columns, bbox, widths, font=13):
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='left', colLoc='left',
                     colWidths=widths, bbox=bbox, edges='horizontal')
    table.auto_set_font_size(False);table.set_fontsize(font)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d8e1eb');cell.set_linewidth(.45)
        cell.set_text_props(color=PALETTE['ink'], fontfamily='DejaVu Sans')
        if row == 0:cell.set_text_props(weight='bold')
    return table


def draw_detailed_frame(fig, axes, seconds, audit_data=None):
    from scripts import screening_animation_audit as audit
    scene, monitor, reference = axes
    state = draw_frame(fig, scene, seconds)
    for axis in (monitor, reference):
        axis.clear();axis.set_xlim(0, 1);axis.set_ylim(0, 1);axis.axis('off')
    summary = audit.frame_statistics(state)
    _text(monitor, 0, .97, 'SCREENING / REPAIR MONITOR', 17, fontweight='bold')
    _text(monitor, 0, .925, 'Toy quantities only — not experiment logs', 12, color=PALETTE['muted'])
    for index, message in enumerate(STEP_NOTES[state.phase]):
        _text(monitor, 0, .875-index*.035, message, 12.5)
    screening = summary['screening']
    strict, expanded = summary['strict'], summary['expanded']
    if state.phase < 5:
        rows = [['Candidate hidden units', str(N_HIDDEN)],
                ['Stable after ranking', str(screening['stable']) if state.phase else 'Not screened'],
                ['Completed ablations', f"{screening['tested']} / {len(STABLE_UNITS)}"],
                ['Passed / failed', f"{screening['passed']} / {screening['failed']}"],
                ['Weight coordinates changed', str(strict['changed_coordinates'])],
                ['Outputs temporarily OFF', str(screening['off'])]]
        _table(monitor, rows, ['Statistic', 'Current value'], [0, .49, .99, .275], [.68, .32], 13)
    else:
        rows = [['Support / cap', f"{strict['editable_coordinates']} / {state.cap}", f"{expanded['editable_coordinates']} / {state.cap}"],
                ['Coordinates changed', str(strict['changed_coordinates']), str(expanded['changed_coordinates'])],
                ['Outside-mask changes', str(strict['frozen_changed_coordinates']), str(expanded['frozen_changed_coordinates'])],
                ['Maximum |Δw|', f"{strict['max_abs_delta']:.5f}", f"{expanded['max_abs_delta']:.5f}"],
                ['RMS Δ inside mask', f"{strict['rms_editable_delta']:.5f}", f"{expanded['rms_editable_delta']:.5f}"],
                ['L2 Δ over all coords', f"{strict['l2_delta']:.5f}", f"{expanded['l2_delta']:.5f}"]]
        _table(monitor, rows, ['Statistic', 'Strict', 'Expanded'], [0, .49, .99, .275], [.48, .26, .26], 12.6)
    _text(monitor, 0, .455, 'UNIT REFERENCE — P / F / not tested', 14, fontweight='bold')
    unit_rows = []
    for row in audit.screening_reference(state):
        outcome = {'not_tested': 'Not tested', 'testing_off': 'OFF (temporary)',
                   'passed': 'P: passed', 'failed': 'F: failed'}[row['ablation']]
        unit_rows.append([row['unit'], f"{row['votes']}/20" if state.phase else '—', outcome,
                          'Selected' if row['strict_selected'] else '—',
                          'Selected' if row['expanded_selected'] else '—'])
    _table(monitor, unit_rows, ['Unit', 'Votes', 'Ablation', 'Strict', 'Expanded'],
           [0, .035, .99, .39], [.12, .17, .29, .21, .21], 12.2)
    _text(monitor, 0, .005, 'Δ values compare with the original toy teacher.', 10.5, color=PALETTE['muted'])
    _text(reference, 0, .96, 'LIVE WEIGHT REFERENCE — stable IDs; complete change history exported to CSV',
          16, fontweight='bold')
    latest = 'E00000 — initial state; no parameters modified.'
    if audit_data is not None and audit_data['events']:
        times = audit_data.get('_event_times')
        if times is None:
            times = [event['time_s'] for event in audit_data['events']]
            audit_data['_event_times'] = times
        index = bisect_right(times, seconds+1e-9)-1
        if index >= 0:
            event = audit_data['events'][index]
            latest = (f"{event['event_id']}  @ {event['time_s']:.2f}s  |  {event['event_type']}"
                      f"  |  {event['policy']}  |  {event['ref']}  |  {event['note']}")
    if len(latest) > 185:latest = latest[:182]+'...'
    _text(reference, 0, .85, latest, 11.5, color=PALETTE['muted'])
    rows = []
    coordinate_rows = audit.coordinate_reference()
    before = base_weights()
    for coordinate in (18, 12, 34, 58):  # H7 input; H5 input; H5 bias; H9 outgoing.
        item = coordinate_rows[coordinate]
        for policy, units in [('Strict', state.strict_units), ('Expanded', state.expanded_units)]:
            values = repaired_weights(units, state.repair_progress)
            delta = float(values[coordinate]-before[coordinate])
            selected = bool(parameter_mask(units)[coordinate])
            status = ('Repaired; frozen' if state.phase >= 7 and selected else
                      'Changed inside mask' if abs(delta)>1e-12 else
                      'Editable; unchanged' if selected else 'Frozen; unchanged')
            rows.append([item['ref'], item['connection'], policy, f'{before[coordinate]:.5f}',
                         f'{values[coordinate]:.5f}', f'{delta:+.5f}', status])
    _table(reference, rows, ['Ref', 'Connection / bias', 'Arm', 'Original', 'Current', 'Δ from original', 'State'],
           [0, .055, 1, .73], [.07, .23, .10, .12, .12, .14, .22], 13)
    _text(reference, 0, .005,
          'P001–P062 cover all coordinates. Δw statistics describe editing magnitude, not accuracy or successful repair.',
          11, color=PALETTE['muted'])
    return state


def _render_signature(state):
    # Hold static explanations without redrawing them. Repair frames still log
    # every sampled coordinate update; no training/event data are skipped.
    motion = state.progress if state.phase in (0, 8) else state.repair_progress if state.phase == 6 else None
    return (state.phase, state.off_unit, state.outcomes, state.cap,
            state.strict_units, state.expanded_units, motion)


def _save_detailed_movie(fig, axes, destination, fps, count, audit_data, ffmpeg):
    fig.canvas.draw();width, height = fig.canvas.get_width_height()
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
           '-vcodec', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{width}x{height}',
           '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
           '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '1', str(destination)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    previous_key = None
    raw = None
    try:
        for frame in range(count):
            seconds = frame/fps;key = _render_signature(state_at(seconds))
            if key != previous_key:
                draw_detailed_frame(fig, axes, seconds, audit_data)
                fig.canvas.draw();raw = np.asarray(fig.canvas.buffer_rgba()).tobytes()
                previous_key = key
            process.stdin.write(raw)
            if frame % max(1, count//10) == 0:
                print(f'  render {100*frame/count:.0f}%', flush=True)
        process.stdin.close()
        error = process.stderr.read().decode(errors='replace')
        if process.wait():raise RuntimeError('ffmpeg export failed: '+error)
    except BaseException:
        process.kill();process.wait()
        raise


def export(output, fps=8, dpi=80, make_gif=True, detailed=True, preview_only=False):
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 30:
        raise ValueError('FPS must be an integer from 1 to 30')
    if (isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 40
            or dpi % (4 if detailed else 2)):
        raise ValueError('DPI must be >=40 and a multiple of 4 for detailed export (compact: even)')
    from matplotlib import animation, pyplot as plt
    from PIL import Image
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None and not preview_only:
        raise RuntimeError('MP4 export needs ffmpeg. Install it separately, or use --show for the Python animation.')
    audit_data = None
    if detailed:
        from scripts import screening_animation_audit as audit
        audit_data = audit.trace(fps)
        audit.export_reports(output, fps, audit_data)
        fig, axes = make_detailed_figure(dpi)
        render = lambda seconds: draw_detailed_frame(fig, axes, seconds, audit_data)
    else:
        fig, ax = make_figure(dpi)
        render = lambda seconds: draw_frame(fig, ax, seconds)
    ends = np.cumsum(DURATIONS)
    preview_time = float(ends[5] + .65 * DURATIONS[6])
    render(preview_time)
    fig.savefig(output / 'screening-repair-preview.png', dpi=dpi, facecolor=PALETTE['bg'])
    # A small six-frame contact sheet supports static inspection without reading a video.
    thumbnails = []
    for time in (.6*DURATIONS[0], float(ends[1]+.3*DURATIONS[2]),
                 float(ends[4]+.9*DURATIONS[5]), preview_time,
                 float(ends[6]+.5*DURATIONS[7]), float(ends[7]+.5*DURATIONS[8])):
        render(time);fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
        frame.thumbnail((640, 360));thumbnails.append(frame)
    sheet = Image.new('RGB', (1280, 1080), PALETTE['bg'])
    for index, frame in enumerate(thumbnails):sheet.paste(frame, ((index % 2)*640, (index//2)*360))
    sheet.save(output / 'screening-repair-storyboard.png')
    if preview_only:
        plt.close(fig)
        print('Saved previews' + (' and reference/statistics tables' if detailed else '')
              + ' to', output.resolve(), flush=True)
        return
    count = math.ceil(TOTAL_SECONDS * fps)
    destination = output / 'screening-repair.mp4'
    print(f'Rendering {count} frames ({TOTAL_SECONDS:.1f}s) on CPU...', flush=True)
    if detailed:
        _save_detailed_movie(fig, axes, destination, fps, count, audit_data, ffmpeg)
    else:
        movie = animation.FuncAnimation(fig, lambda frame: render(frame/fps),
                                        frames=count, interval=1000/fps, repeat=False, cache_frame_data=False)
        writer = animation.FFMpegWriter(fps=fps, codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-crf', '20', '-threads', '1'])
        movie.save(str(destination), writer=writer, dpi=dpi)
    plt.close(fig)
    if make_gif:
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-threads', '1',
                        '-i', str(destination), '-filter_complex_threads', '1', '-filter_complex',
                        f'fps={min(fps, 6)},scale={1440 if detailed else 1024}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                        '-loop', '-1', str(output / 'screening-repair.gif')], check=True)
    files = {}
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == 'animation-manifest.json':continue
        with path.open('rb') as handle:digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest}
    (output / 'animation-manifest.json').write_text(json.dumps({
        'conceptual_only': True, 'training_performed': False, 'toy_hidden_units': N_HIDDEN,
        'toy_parameter_count': TOTAL_PARAMETERS, 'duration_seconds': TOTAL_SECONDS,
        'fps': fps, 'frames': count, 'detailed': detailed, 'files': files,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_source_sha256': (hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
                                if detailed else None)}, indent=2)+'\n')
    print('Saved animation and previews to', output.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--dpi', type=int, default=80)
    parser.add_argument('--no-gif', action='store_true')
    parser.add_argument('--show', action='store_true', help='Play interactively instead of exporting')
    parser.add_argument('--compact', action='store_true', help='Original short, diagram-only presentation')
    parser.add_argument('--preview-only', action='store_true', help='Export reference tables and static previews only')
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or args.dpi < 40 or args.dpi % (2 if args.compact else 4):
        parser.error('Use fps 1–30 and dpi >=40; detailed dpi must be a multiple of 4 (compact: even)')
    if args.output is None:
        args.output = Path('runs/visualizations/screening-repair' + ('' if args.compact else '-detailed'))
    if args.compact:
        global DURATIONS, TOTAL_SECONDS
        DURATIONS = COMPACT_DURATIONS;TOTAL_SECONDS = sum(DURATIONS)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve() / '.matplotlib-cache'))
    import matplotlib
    if not args.show:matplotlib.use('Agg')
    if args.show:
        from matplotlib import animation, pyplot as plt
        if args.compact:
            fig, ax = make_figure(args.dpi)
            render = lambda seconds: draw_frame(fig, ax, seconds)
        else:
            from scripts import screening_animation_audit as audit
            data = audit.trace(args.fps)
            fig, axes = make_detailed_figure(args.dpi)
            render = lambda seconds: draw_detailed_frame(fig, axes, seconds, data)
        movie = animation.FuncAnimation(fig, lambda frame: render(frame/args.fps),
            frames=math.ceil(TOTAL_SECONDS*args.fps), interval=1000/args.fps,
            repeat=False, cache_frame_data=False)
        plt.show()
        return movie
    export(args.output, args.fps, args.dpi, not args.no_gif, not args.compact, args.preview_only)


if __name__ == '__main__':main()
