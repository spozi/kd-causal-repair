"""Computed statistics and a lossless sampled change log for the teaching animation.

All values refer to ``animate_screening_repair``'s illustrative ten-unit MLP.
This module does not train a model, measure accuracy/loss/KL, open experiment
data or contact a GPU server. Numerical deltas are computed from the actual
displayed toy weight arrays, not invented performance improvements.
"""

import csv
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from scripts import animate_screening_repair as story


POLICIES = ('strict', 'expanded')
_TOLERANCE = 1e-12


def _validated_fps(fps):
    if isinstance(fps, bool) or not isinstance(fps, Integral) or not 1 <= fps <= 30:
        raise ValueError('Audit fps must be an integer from 1 to 30')
    return int(fps)


def coordinate_reference():
    """Stable P001..P062 IDs for every toy parameter, including frozen biases."""
    before = story.base_weights()
    rows = []
    incoming_count = story.N_INPUT * story.N_HIDDEN
    bias_end = incoming_count + story.N_HIDDEN
    outgoing_end = bias_end + story.N_HIDDEN * story.N_OUTPUT
    for index in range(story.TOTAL_PARAMETERS):
        if index < incoming_count:
            unit, source = divmod(index, story.N_INPUT)
            owner = f'H{unit + 1}'
            kind = 'incoming_weight'
            connection = f'x{source + 1} -> {owner}'
        elif index < bias_end:
            unit = index - incoming_count
            owner = f'H{unit + 1}'
            kind = 'hidden_bias'
            connection = f'bias -> {owner}'
        elif index < outgoing_end:
            output, unit = divmod(index - bias_end, story.N_HIDDEN)
            owner = f'H{unit + 1}'
            kind = 'outgoing_weight'
            connection = f'{owner} -> y{output + 1}'
        else:
            output = index - outgoing_end
            owner = None
            kind = 'output_bias'
            connection = f'bias -> y{output + 1}'
        rows.append({'ref': f'P{index + 1:03d}', 'index': index,
                     'owner_unit': owner, 'connection': connection, 'kind': kind,
                     'baseline': float(before[index])})
    return rows


def screening_reference(state):
    """Per-unit votes, hypothetical ablation status and current mask eligibility."""
    outcomes = dict(state.outcomes)
    assessed = state.phase >= 1
    rows = []
    for unit in range(story.N_HIDDEN):
        outcome = outcomes.get(unit)
        ablation = ('testing_off' if state.off_unit == unit else
                    'passed' if outcome == 'P' else
                    'failed' if outcome == 'F' else 'not_tested')
        votes = int(story.BOOTSTRAP_VOTES[unit]) if assessed else None
        rows.append({'unit': f'H{unit + 1}', 'votes': votes,
                     'stability': votes / 20 if votes is not None else None,
                     'ablation': ablation,
                     'strict_selected': unit in state.strict_units,
                     'expanded_selected': unit in state.expanded_units})
    return rows


def _policy_statistics(state, policy):
    units = getattr(state, f'{policy}_units')
    mask = story.parameter_mask(units)
    delta = story.repaired_weights(units, state.repair_progress) - story.base_weights()
    frozen_changed = int(np.count_nonzero(np.abs(delta[~mask]) > _TOLERANCE))
    if frozen_changed:
        raise ValueError('Illustration changed a frozen coordinate')
    editable = int(mask.sum())
    return {'selected_units': len(units), 'editable_coordinates': editable,
            'changed_coordinates': int(np.count_nonzero(np.abs(delta) > _TOLERANCE)),
            'frozen_changed_coordinates': frozen_changed,
            'max_abs_delta': float(np.max(np.abs(delta))),
            'l2_delta': float(np.linalg.norm(delta)),
            'rms_editable_delta': float(np.sqrt(np.mean(delta[mask] ** 2))) if editable else 0.0,
            'cap': int(state.cap)}


def frame_statistics(state):
    """Counts and true array-derived deltas; never model-performance statistics."""
    rows = screening_reference(state)
    passed = sum(row['ablation'] == 'passed' for row in rows)
    failed = sum(row['ablation'] == 'failed' for row in rows)
    return {'screening': {'candidates': story.N_HIDDEN,
                          'stable': len(story.STABLE_UNITS) if state.phase >= 1 else 0,
                          'tested': passed + failed, 'passed': passed, 'failed': failed,
                          'not_tested': sum(row['ablation'] == 'not_tested' for row in rows),
                          'off': sum(row['ablation'] == 'testing_off' for row in rows)},
            **{policy: _policy_statistics(state, policy) for policy in POLICIES}}


def trace(fps):
    """Log every state and scalar weight change at the exported video sample rate.

    Ablation records are symbolic activation interventions, not changes to any
    weight coordinate. Initial values live in ``coordinate_reference``. Summing
    each Pxxx reference's ``weight_change`` deltas reconstructs its final value.
    """
    fps = _validated_fps(fps)
    reference = coordinate_reference()
    events, frames = [], []
    previous = None
    previous_weights = {policy: story.base_weights() for policy in POLICIES}

    def event(time, phase, event_type, policy='', ref='', connection='', before='',
              after='', delta='', note=''):
        events.append({'event_id': f'E{len(events) + 1:05d}', 'time_s': float(time),
                       'phase': int(phase), 'event_type': event_type, 'policy': policy,
                       'ref': ref, 'connection': connection, 'before': before,
                       'after': after, 'delta': delta, 'note': note})

    for frame in range(math.ceil(story.TOTAL_SECONDS * fps)):
        time = frame / fps
        state = story.state_at(time)
        stats = frame_statistics(state)
        if previous is None or state.phase != previous.phase:
            event(time, state.phase, 'stage_change', ref=f'STAGE.{state.phase + 1}',
                  before=story.PHASE_TITLES[previous.phase] if previous else 'not_started',
                  after=story.PHASE_TITLES[state.phase], note='Conceptual animation stage; not a training epoch')
        if previous is not None and previous.off_unit != state.off_unit and previous.off_unit is not None:
            unit = previous.off_unit
            event(time, state.phase, 'activation_restore', policy='shared', ref=f'ACT.H{unit + 1}',
                  connection=f'output of H{unit + 1}', before='forced_zero', after='original_output',
                  note='Temporary ablation hook removed; all weights stayed unchanged')
        if state.off_unit is not None and (previous is None or previous.off_unit != state.off_unit):
            unit = state.off_unit
            event(time, state.phase, 'activation_off', policy='shared', ref=f'ACT.H{unit + 1}',
                  connection=f'output of H{unit + 1}', before='original_output', after='forced_zero',
                  note='Symbolic output intervention only; no numeric activation value is invented')
        if state.phase >= 1 and (previous is None or previous.phase < 1):
            for row in screening_reference(state):
                unit = int(row['unit'][1:]) - 1
                event(time, state.phase, 'ranking_assessed', policy='shared', ref=f'SCREEN.{row["unit"]}',
                      connection=row['unit'], before='unassessed',
                      after='stable' if unit in story.STABLE_UNITS else 'not_stable',
                      note=f'Invented toy vote count: {row["votes"]}/20; stability={row["stability"]:.2f}')
        old_outcomes = dict(previous.outcomes) if previous else {}
        for unit, outcome in state.outcomes:
            if old_outcomes.get(unit) != outcome:
                event(time, state.phase, 'screening_outcome', policy='shared', ref=f'SCREEN.H{unit + 1}',
                      connection=f'H{unit + 1}', before='not_tested',
                      after='passed' if outcome == 'P' else 'failed',
                      note='Hypothetical ablation outcome, not measured experimental efficacy')
        for policy in POLICIES:
            units = getattr(state, f'{policy}_units')
            mask = story.parameter_mask(units)
            previous_mask = (story.parameter_mask(getattr(previous, f'{policy}_units'))
                             if previous is not None else np.zeros(story.TOTAL_PARAMETERS, dtype=bool))
            for index in np.flatnonzero(mask != previous_mask):
                ref = reference[int(index)]
                event(time, state.phase, 'support_change', policy=policy, ref=ref['ref'],
                      connection=ref['connection'], before=bool(previous_mask[index]), after=bool(mask[index]),
                      note='Editable-support eligibility changed; this event itself does not update the weight')
            weights = story.repaired_weights(units, state.repair_progress)
            changes = weights - previous_weights[policy]
            for index in np.flatnonzero(np.abs(changes) > _TOLERANCE):
                if not mask[index]:
                    raise ValueError('A logged weight update is outside the editable support')
                ref = reference[int(index)]
                event(time, state.phase, 'weight_change', policy=policy, ref=ref['ref'],
                      connection=ref['connection'], before=float(previous_weights[policy][index]),
                      after=float(weights[index]), delta=float(changes[index]),
                      note='Illustrative interpolation sample; not SGD and not a measured model statistic')
            previous_weights[policy] = weights
        flattened = {'frame_index': frame, 'time_s': float(time), 'phase': state.phase,
                     'phase_title': story.PHASE_TITLES[state.phase],
                     'repair_progress': state.repair_progress, 'event_count': len(events)}
        for group, values in stats.items():
            flattened.update({f'{group}_{key}': value for key, value in values.items()})
        frames.append(flattened)
        previous = state
    return {'fps': fps, 'duration_seconds': story.TOTAL_SECONDS,
            'events': events, 'frames': frames}


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Audit exports must not silently omit an empty table')
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def export_reports(output, fps, trace_data=None):
    """Write full-coordinate reference/final tables, sampled events and statistics."""
    fps = _validated_fps(fps)
    data = trace(fps) if trace_data is None else trace_data
    if data['fps'] != fps or data['duration_seconds'] != story.TOTAL_SECONDS:
        raise ValueError('Trace sampling or story duration differs from the requested export')
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    reference = coordinate_reference()
    final = story.state_at(story.TOTAL_SECONDS)
    comparison = []
    for policy in POLICIES:
        units = getattr(final, f'{policy}_units')
        mask = story.parameter_mask(units)
        weights = story.repaired_weights(units, final.repair_progress)
        for row in reference:
            index = row['index']
            comparison.append({'policy': policy, **row, 'editable': bool(mask[index]),
                               'final': float(weights[index]),
                               'delta': float(weights[index] - row['baseline']),
                               'changed': bool(abs(weights[index] - row['baseline']) > _TOLERANCE)})
    tables = {'coordinate-reference.csv': reference,
              'unit-screening-reference.csv': screening_reference(final),
              'final-weight-comparison.csv': comparison,
              'change-log.csv': data['events'], 'frame-statistics.csv': data['frames']}
    for name, rows in tables.items():
        _write_csv(output / name, rows)
    summary = {'conceptual_only': True, 'training_performed': False,
               'real_model_metrics_evaluated': False, 'accuracy_loss_kl_traces_invented': False,
               'weight_motion': 'Deterministic toy interpolation; not an optimizer simulation',
               'screening_values': 'Invented teaching outcomes/votes; not experiment results',
               'activation_interventions': 'Symbolic original_output/forced_zero; no invented numeric activation',
               'fps': fps, 'duration_seconds': story.TOTAL_SECONDS,
               'coordinate_references': len(reference), 'policy_coordinate_rows': len(comparison),
               'unit_screening_references': story.N_HIDDEN,
               'sampled_frames': len(data['frames']), 'logged_events': len(data['events']),
               'weight_change_events': sum(row['event_type'] == 'weight_change' for row in data['events']),
               'final_statistics': frame_statistics(final),
               'definitions': {'changed_coordinates': 'abs(current-original)>1e-12',
                               'max_abs_delta': 'max(abs(current-original)) over all 62 coordinates',
                               'l2_delta': 'Euclidean norm(current-original)',
                               'rms_editable_delta': 'sqrt(mean(delta^2)) over editable coordinates',
                               'not_tested': 'No completed ablation and not currently forced off',
                               'event_time': 'Video sample time, not a wall-clock/training time'}}
    (output / 'audit-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return {name: str(output / name) for name in [*tables, 'audit-summary.json']}
