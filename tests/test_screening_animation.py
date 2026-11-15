"""Conceptual animation invariants, not experimental efficacy tests."""
import math

import numpy as np
import pytest

from scripts import animate_screening_repair as story


def start(phase):
    return sum(story.DURATIONS[:phase])


def test_toy_topology_and_bootstrap_votes_are_internally_consistent():
    assert story.TOTAL_PARAMETERS == (story.N_INPUT*story.N_HIDDEN + story.N_HIDDEN
                                    + story.N_HIDDEN*story.N_OUTPUT + story.N_OUTPUT)
    assert sum(story.BOOTSTRAP_VOTES) == math.ceil(story.N_HIDDEN*.25)*20
    assert tuple(index for index, count in enumerate(story.BOOTSTRAP_VOTES) if count/20 >= .7) == story.STABLE_UNITS
    assert set(story.STRICT_RANKING) == {index for index, outcome in story.TEST_OUTCOMES.items() if outcome=='P'}
    assert set(story.STRICT_RANKING + story.FALLBACK_RANKING) == set(range(story.N_HIDDEN))


def test_equal_toy_budget_strict_saturates_expanded_uses_fallback():
    strict = story.select_units('strict', 24)
    expanded = story.select_units('expanded', 24)
    assert strict == story.STRICT_RANKING
    assert expanded == story.STRICT_RANKING + story.FALLBACK_RANKING[:2]
    assert story.parameter_mask(strict).sum() == 12
    assert story.parameter_mask(expanded).sum() == 24
    assert story.select_units('expanded', 12) == strict
    assert story.select_units('strict', 5) == ()


@pytest.mark.parametrize('units', [story.STRICT_RANKING, story.STRICT_RANKING+story.FALLBACK_RANKING[:2]])
def test_only_selected_weight_coordinates_change_and_network_does_not_grow(units):
    before = story.base_weights()
    after = story.repaired_weights(units, .7)
    mask = story.parameter_mask(units)
    assert before.shape == after.shape == (62,)
    assert np.array_equal(before[~mask], after[~mask])
    assert np.any(before[mask] != after[mask])
    # Output biases are outside all toy hidden-unit masks.
    assert not mask[-2:].any()
    assert np.array_equal(story.repaired_weights(units, 0), before)


@pytest.mark.parametrize('phase,unit', [(2, 6), (3, 4), (4, 2)])
def test_ablation_is_temporary_and_does_not_update_weights(phase, unit):
    off = story.state_at(start(phase)+.25*story.DURATIONS[phase])
    restored = story.state_at(start(phase)+.8*story.DURATIONS[phase])
    assert off.off_unit == unit
    assert restored.off_unit is None
    assert (unit, story.TEST_OUTCOMES[unit]) in restored.outcomes
    assert off.repair_progress == restored.repair_progress == 0
    assert off.strict_units == off.expanded_units == ()


def test_same_budget_increases_without_changing_screening_outcomes():
    states = [story.state_at(start(5)+fraction*story.DURATIONS[5]) for fraction in (.1, .5, .9)]
    assert [state.cap for state in states] == [12, 18, 24]
    assert [len(state.strict_units) for state in states] == [2, 2, 2]
    assert [len(state.expanded_units) for state in states] == [2, 3, 4]
    assert all(state.repair_progress == 0 for state in states)
    assert all(dict(state.outcomes) == story.TEST_OUTCOMES for state in states)


def test_phase_boundaries_and_final_frame_are_stable():
    assert story.state_at(0).phase == 0
    assert story.state_at(start(6)+2).phase == 6
    assert story.state_at(story.TOTAL_SECONDS).phase == 8
    assert story.state_at(story.TOTAL_SECONDS+5) == story.state_at(story.TOTAL_SECONDS)
    for invalid in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):story.state_at(invalid)


def test_every_scene_renders_headlessly_without_training_or_external_resources(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = story.make_figure(dpi=50)
    for phase in range(len(story.DURATIONS)):
        for fraction in ((.2, .8) if phase in (2, 3, 4) else (.7,)):
            state = story.draw_frame(fig, ax, start(phase)+fraction*story.DURATIONS[phase])
            assert state.phase == phase
            fig.canvas.draw()
    fig.savefig(tmp_path/'preview.png')
    assert (tmp_path/'preview.png').stat().st_size > 1000
    plt.close(fig)


def test_verbose_monitor_and_reference_tables_fit_without_clipping():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from scripts import screening_animation_audit as audit
    data = audit.trace(2)
    fig, axes = story.make_detailed_figure(dpi=80)
    for phase in (0, 1, 2, 5, 6, 7, 8):
        story.draw_detailed_frame(fig, axes, start(phase)+.7*story.DURATIONS[phase], data)
        fig.canvas.draw();renderer = fig.canvas.get_renderer()
        for axis in axes[1:]:
            for table in axis.tables:
                for cell in table.get_celld().values():
                    text = cell.get_text()
                    if not text.get_text():continue
                    box, content = cell.get_window_extent(renderer), text.get_window_extent(renderer)
                    assert content.x0 >= box.x0 - 1
                    assert content.x1 <= box.x1 + 1, text.get_text()
                    assert content.y0 >= box.y0 - 1
                    assert content.y1 <= box.y1 + 1, text.get_text()
    plt.close(fig)


def test_static_frame_caching_does_not_hide_visible_state_changes():
    signature = story._render_signature
    assert signature(story.state_at(start(1)+1)) == signature(story.state_at(start(1)+2))
    assert signature(story.state_at(start(2)+.2)) != signature(story.state_at(start(2)+5))
    assert signature(story.state_at(start(6)+1)) != signature(story.state_at(start(6)+2))
    assert signature(story.state_at(start(6)+19)) == signature(story.state_at(start(6)+19.5))


def test_verbose_preview_exports_full_audit_without_ffmpeg(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(story.shutil, 'which', lambda name: None)
    story.export(tmp_path, fps=2, dpi=40, preview_only=True)
    assert (tmp_path/'screening-repair-preview.png').is_file()
    assert (tmp_path/'unit-screening-reference.csv').is_file()
    assert (tmp_path/'coordinate-reference.csv').is_file()
    assert (tmp_path/'change-log.csv').is_file()
    summary = json.loads((tmp_path/'audit-summary.json').read_text())
    assert summary['unit_screening_references'] == 10
    assert summary['coordinate_references'] == 62
    assert summary['policy_coordinate_rows'] == 124
    assert not summary['real_model_metrics_evaluated']
    assert not (tmp_path/'screening-repair.mp4').exists()


@pytest.mark.parametrize('kwargs', [{'fps': 0}, {'fps': True}, {'fps': 31}, {'fps': 2.5},
                                   {'dpi': 50}, {'dpi': 39}, {'dpi': True}])
def test_export_api_rejects_invalid_sampling_or_odd_h264_dimensions_before_writing(tmp_path, kwargs):
    destination = tmp_path/'invalid'
    with pytest.raises(ValueError):story.export(destination, **kwargs)
    assert not destination.exists()
