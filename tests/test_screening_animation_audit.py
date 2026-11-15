"""Audit tables must describe the toy arrays faithfully, never model efficacy."""
import csv
import json
import math

import numpy as np
import pytest

from scripts import animate_screening_repair as story
from scripts import screening_animation_audit as audit


def point(phase, fraction=.7):
    return sum(story.DURATIONS[:phase]) + fraction * story.DURATIONS[phase]


def test_coordinate_reference_is_complete_stable_and_maps_displayed_examples():
    rows = audit.coordinate_reference()
    assert len(rows) == 62
    assert len({row['ref'] for row in rows}) == 62
    assert [row['index'] for row in rows] == list(range(62))
    assert rows == audit.coordinate_reference()
    assert rows[18]['ref'] == 'P019' and rows[18]['connection'] == 'x1 -> H7'
    assert rows[12]['ref'] == 'P013' and rows[12]['connection'] == 'x1 -> H5'
    assert rows[34]['ref'] == 'P035' and rows[34]['connection'] == 'bias -> H5'
    assert rows[58]['ref'] == 'P059' and rows[58]['connection'] == 'H9 -> y2'
    assert rows[-1]['owner_unit'] is None and rows[-1]['kind'] == 'output_bias'
    assert np.array_equal([row['baseline'] for row in rows], story.base_weights())


def test_before_assessment_has_no_invented_votes_or_screening_outcomes():
    state = story.state_at(0)
    rows = audit.screening_reference(state)
    assert all(row['votes'] is None and row['stability'] is None for row in rows)
    assert all(row['ablation'] == 'not_tested' for row in rows)
    stats = audit.frame_statistics(state)
    assert stats['screening'] == {'candidates': 10, 'stable': 0, 'tested': 0,
                                  'passed': 0, 'failed': 0, 'not_tested': 10, 'off': 0}
    assert stats['strict']['changed_coordinates'] == stats['expanded']['changed_coordinates'] == 0


def test_ablation_off_is_an_activation_intervention_not_a_weight_change():
    state = story.state_at(point(2, .2))
    rows = audit.screening_reference(state)
    assert rows[6]['ablation'] == 'testing_off'
    stats = audit.frame_statistics(state)
    assert stats['screening']['off'] == 1
    assert stats['screening']['tested'] == 0
    assert stats['screening']['not_tested'] == 9
    assert all(stats[policy]['changed_coordinates'] == 0 for policy in audit.POLICIES)
    restored = story.state_at(point(2, .8))
    assert audit.screening_reference(restored)[6]['ablation'] == 'passed'


@pytest.mark.parametrize('policy,selected,coordinates', [('strict', 2, 12), ('expanded', 4, 24)])
def test_repair_statistics_are_computed_from_actual_arrays(policy, selected, coordinates):
    state = story.state_at(point(6, .5))
    stats = audit.frame_statistics(state)[policy]
    units = getattr(state, f'{policy}_units')
    mask = story.parameter_mask(units)
    delta = story.repaired_weights(units, state.repair_progress) - story.base_weights()
    assert stats['selected_units'] == selected
    assert stats['editable_coordinates'] == stats['changed_coordinates'] == coordinates
    assert stats['frozen_changed_coordinates'] == 0
    assert stats['max_abs_delta'] == float(np.abs(delta).max())
    assert stats['l2_delta'] == float(np.linalg.norm(delta))
    assert stats['rms_editable_delta'] == float(np.sqrt(np.mean(delta[mask] ** 2)))


def test_final_screening_reference_distinguishes_failed_and_untested_fallback():
    state = story.state_at(story.TOTAL_SECONDS)
    rows = {row['unit']: row for row in audit.screening_reference(state)}
    assert rows['H5']['ablation'] == 'failed'
    assert rows['H9']['ablation'] == 'not_tested'
    assert not rows['H5']['strict_selected'] and rows['H5']['expanded_selected']
    assert not rows['H9']['strict_selected'] and rows['H9']['expanded_selected']
    stats = audit.frame_statistics(state)['screening']
    assert (stats['stable'], stats['tested'], stats['passed'], stats['failed'], stats['not_tested']) == (3, 3, 2, 1, 7)


def test_statistics_fail_closed_if_a_frozen_output_bias_changes(monkeypatch):
    original = story.repaired_weights
    def invalid(units, progress):
        values = original(units, progress)
        values[-1] += .01
        return values
    monkeypatch.setattr(story, 'repaired_weights', invalid)
    with pytest.raises(ValueError, match='frozen coordinate'):
        audit.frame_statistics(story.state_at(point(6)))


def test_sampled_change_events_reconstruct_every_final_weight_and_never_change_frozen_coordinates():
    data = audit.trace(2)
    assert data == audit.trace(2)
    assert len(data['frames']) == math.ceil(story.TOTAL_SECONDS * 2)
    events = data['events']
    assert [row['event_id'] for row in events] == [f'E{i+1:05d}' for i in range(len(events))]
    assert [row['time_s'] for row in events] == sorted(row['time_s'] for row in events)
    current = {policy: story.base_weights().copy() for policy in audit.POLICIES}
    for event in events:
        if event['event_type'] != 'weight_change':
            continue
        index = int(event['ref'][1:]) - 1
        state = story.state_at(event['time_s'])
        mask = story.parameter_mask(getattr(state, f'{event["policy"]}_units'))
        assert mask[index]
        assert event['before'] == pytest.approx(current[event['policy']][index], abs=1e-14)
        assert event['after'] - event['before'] == pytest.approx(event['delta'], abs=1e-14)
        current[event['policy']][index] += event['delta']
    final = story.state_at(story.TOTAL_SECONDS)
    for policy in audit.POLICIES:
        expected = story.repaired_weights(getattr(final, f'{policy}_units'), final.repair_progress)
        assert np.allclose(current[policy], expected, atol=1e-14, rtol=0)


def test_all_temporary_activations_are_restored_and_symbolic_only():
    events = audit.trace(2)['events']
    offs = [row for row in events if row['event_type'] == 'activation_off']
    restores = [row for row in events if row['event_type'] == 'activation_restore']
    assert len(offs) == len(restores) == 3
    assert {row['ref'] for row in offs} == {row['ref'] for row in restores} == {'ACT.H7', 'ACT.H5', 'ACT.H3'}
    assert all(row['before'] == 'original_output' and row['after'] == 'forced_zero' and row['delta'] == '' for row in offs)
    assert all(row['before'] == 'forced_zero' and row['after'] == 'original_output' and row['delta'] == '' for row in restores)
    assert not any(row['event_type'] == 'weight_change' and row['phase'] <= 5 for row in events)


def test_repair_reaches_final_weights_before_validation_and_then_stays_frozen():
    events = audit.trace(2)['events']
    assert all(row['phase'] == 6 for row in events if row['event_type'] == 'weight_change')
    late_repair = story.state_at(point(6, .95))
    assert late_repair.repair_progress == 1.0
    for phase in (7, 8):
        later = story.state_at(point(phase, .5))
        for policy in audit.POLICIES:
            units = getattr(late_repair, f'{policy}_units')
            assert np.array_equal(story.repaired_weights(units, late_repair.repair_progress),
                                  story.repaired_weights(getattr(later, f'{policy}_units'), later.repair_progress))


def test_exported_tables_have_complete_references_histories_and_disclaimers(tmp_path):
    data = audit.trace(2)
    paths = audit.export_reports(tmp_path, 2, data)
    assert set(paths) == {'coordinate-reference.csv', 'unit-screening-reference.csv',
                          'final-weight-comparison.csv', 'change-log.csv',
                          'frame-statistics.csv', 'audit-summary.json'}
    def read(name):
        with open(paths[name], newline='') as handle:
            return list(csv.DictReader(handle))
    assert len(read('coordinate-reference.csv')) == 62
    screening = read('unit-screening-reference.csv')
    assert len(screening) == 10
    assert {row['unit'] for row in screening} == {f'H{index + 1}' for index in range(10)}
    assert sum(row['ablation']=='passed' for row in screening) == 2
    assert sum(row['ablation']=='failed' for row in screening) == 1
    assert sum(row['ablation']=='not_tested' for row in screening) == 7
    comparison = read('final-weight-comparison.csv')
    assert len(comparison) == 124
    assert sum(row['changed']=='True' for row in comparison if row['policy']=='strict') == 12
    assert sum(row['changed']=='True' for row in comparison if row['policy']=='expanded') == 24
    assert not any(row['changed']=='True' and row['editable']=='False' for row in comparison)
    assert len(read('change-log.csv')) == len(data['events'])
    assert len(read('frame-statistics.csv')) == len(data['frames'])
    summary = json.loads(open(paths['audit-summary.json']).read())
    assert summary['conceptual_only'] and not summary['training_performed']
    assert summary['unit_screening_references'] == 10
    assert not summary['real_model_metrics_evaluated']
    assert summary['final_statistics']['strict']['frozen_changed_coordinates'] == 0
    assert summary['final_statistics']['expanded']['frozen_changed_coordinates'] == 0


@pytest.mark.parametrize('fps', [0, -1, 31, True, 2.5, float('nan'), '8'])
def test_invalid_audit_fps_is_rejected(fps):
    with pytest.raises(ValueError):
        audit.trace(fps)


def test_export_rejects_inconsistent_trace_sampling(tmp_path):
    with pytest.raises(ValueError, match='Trace sampling'):
        audit.export_reports(tmp_path, 2, audit.trace(1))
