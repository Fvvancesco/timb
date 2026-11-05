import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from icon_pokemon.expanded_search import (
    candidates, is_restricted, row_indices, scale_pairs, swap, training_data,
)
from scripts.run_expanded_search import atomic, checkpoint


def test_full_cartesian_grids_include_all_original_choices():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'config/expanded_search_v1.json').read_text())
    grid = candidates(config)
    assert {name: sum(c['family'] == name for c in grid) for name in config['grids']} == {
        'logistic': 5, 'boosting': 81, 'mlp': 81}
    assert all(sum(c['family'] == name and is_restricted(c) for c in grid) == 3 for name in config['grids'])
    assert len({json.dumps(c, sort_keys=True) for c in grid}) == len(grid)


def test_split_excludes_crossing_matches_before_augmentation():
    examples = [('a', 'b', 1), ('b', 'a', 0), ('c', 'd', 1), ('a', 'c', 0), ('x', 'y', 1)]
    split = row_indices(examples, {'a', 'b'}, {'c', 'd'})
    assert split == {'train': [0, 1], 'validation': [2], 'discarded': 2}
    with pytest.raises(ValueError, match='Overlapping team'):
        row_indices(examples, {'a'}, {'a', 'b'})


def test_augmentation_preserves_total_weight_and_complements_labels():
    x = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    y = np.array([1, 0])
    xx, yy, weights = training_data(x, y, True)
    np.testing.assert_array_equal(xx, [[1, 2, 3, 4], [5, 6, 7, 8], [3, 4, 1, 2], [7, 8, 5, 6]])
    np.testing.assert_array_equal(yy, [1, 0, 0, 1])
    assert weights.sum() == len(y)
    assert np.sum(weights[yy == 1]) == np.sum(weights[yy == 0])
    np.testing.assert_array_equal(training_data(x, y, False)[0], x)


def test_common_scaler_commutes_with_swap_and_uses_only_fit_rows():
    train = np.array([[0., 2., 4., 6.], [2., 4., 6., 8.]])
    a, b = np.hsplit(train, 2)
    scaler = StandardScaler().fit(np.vstack((a, b)))
    np.testing.assert_allclose(scaler.mean_, [3, 5])
    valid = np.array([[100., 200., -100., -200.]])
    np.testing.assert_allclose(scale_pairs(scaler, swap(valid)), swap(scale_pairs(scaler, valid)))
    np.testing.assert_allclose(scaler.mean_, [3, 5])
    with pytest.raises(ValueError, match='equally sized'):
        swap(np.zeros((2, 3)))


def test_resume_rejects_wrong_protocol_or_candidate(tmp_path):
    file = tmp_path / 'checkpoint.json'
    assert checkpoint(file, 'digest', candidate={'C': 1}) is None
    atomic(file, {'protocol_sha256': 'digest', 'candidate': {'C': 1}})
    assert checkpoint(file, 'digest', candidate={'C': 1})['candidate'] == {'C': 1}
    with pytest.raises(ValueError, match='Invalid checkpoint'):
        checkpoint(file, 'other', candidate={'C': 1})
    with pytest.raises(ValueError, match='Invalid checkpoint'):
        checkpoint(file, 'digest', candidate={'C': .1})
