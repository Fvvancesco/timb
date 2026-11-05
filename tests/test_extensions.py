"""Small independent checks of completion semantics, without any training."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_team_completion_extension import bayes, freeze, input_matrix, masked_cases, score
from run_team_profile_extension import profile_matrix
from run_completion_solver_extension import scenario_weights
from audit_extensions import compositions


def test_six_masks_have_no_target_in_visible_and_preserve_unknowns():
    visible, targets = masked_cases([list('abcdef')], list('abcde'))
    assert visible.shape == (6, 5)
    assert targets.tolist() == [0, 1, 2, 3, 4, -1]
    assert all(target not in row for row, target in zip(visible, targets, strict=True))
    assert (visible == -1).sum() == 5


def test_bag_is_order_invariant_and_unknown_does_not_become_last_species():
    visible = np.array([[0, 1, 2, 3, -1]])
    matrix = input_matrix(visible, 6).toarray()
    np.testing.assert_allclose(matrix, [[.2, .2, .2, .2, 0, 0]])
    np.testing.assert_equal(matrix, input_matrix(visible[:, ::-1], 6).toarray())


def test_bayes_matches_hand_calculated_positive_evidence_posterior():
    # Observe species 0. Unnormalized weights for hidden 1 and 2 are
    # (3/10)*(1+1)/(3+2) = .12 and (2/10)*(2+1)/(2+2) = .15.
    counts = np.array([5, 3, 2])
    cooccurrence = np.array([[0, 1, 2], [1, 0, 0], [2, 0, 0]])
    visible = np.array([[0, -1]])
    np.testing.assert_allclose(bayes(visible, counts, cooccurrence, 1), [[0, 4/9, 5/9]])
    np.testing.assert_allclose(bayes(visible, counts, cooccurrence), [[0, .6, .4]])


def test_unknown_target_is_topk_miss_and_excluded_only_from_logloss():
    probabilities = np.array([[.4, .3, .2, .06, .03, .01]] * 2)
    _, scores = score(probabilities, np.array([0, -1]))
    np.testing.assert_allclose(scores[0], [1, 1, -np.log(.4)])
    assert scores[1, :2].tolist() == [0, 0]
    assert np.isnan(scores[1, 2])


def test_profile_average_keeps_missing_members_and_availability_explicit():
    visible = np.array([[0, 1, 2, -1, 3]])
    profiles = {'a': {'kb': [1., 0.]}, 'c': {'kb': [0., .5]}}
    result = profile_matrix(visible, list('abcd'), profiles, 'kb').toarray()
    np.testing.assert_allclose(result[0, 4:], [.2, .1, .4])
    np.testing.assert_allclose(result, profile_matrix(visible[:, ::-1], list('abcd'), profiles, 'kb').toarray())


def test_scenario_quantization_keeps_budget_and_low_probability_scenarios():
    assert scenario_weights([('a', .3)]) == [('a', 20)]
    weights = scenario_weights(list(zip('abcde', [.9, .05, .03, .019, .001], strict=True)))
    assert sum(n for _, n in weights) == 20
    assert all(n >= 1 for _, n in weights)
    assert weights[0][1] == max(n for _, n in weights)


def test_freeze_rejects_changed_artifact_and_keeps_original(tmp_path):
    path = tmp_path / 'selection.json'
    freeze(path, {'dimension': 32})
    original = path.read_bytes()
    freeze(path, {'dimension': 32})
    with pytest.raises(ValueError, match='Frozen artifact differs'):
        freeze(path, {'dimension': 16})
    assert path.read_bytes() == original


def test_audit_detects_reordered_duplicate_composition():
    with pytest.raises(ValueError, match='Repeated unordered'):
        compositions([list('abcdef'), list('fedcba')])
