import numpy as np
import pytest
from icon_pokemon.evaluation import grouped_interval, probability_losses, paired_sign_flip


def test_known_probabilities_and_zero_paired_differences():
    losses = probability_losses([1, 0], [.5, .5])
    assert losses[:, 0] == pytest.approx([np.log(2), np.log(2)])
    assert losses[:, 1] == pytest.approx([.25, .25])
    assert losses[:, 2].tolist() == [1, 0]
    result = grouped_interval(losses - losses, ['a', 'b'], repetitions=100)
    assert result['mean'] == [0, 0, 0]
    assert result['interval_95'] == [[0, 0], [0, 0], [0, 0]]


def test_cluster_resampling_keeps_replicates_and_reports_missing_information():
    result = grouped_interval([0, 0, 1, 1], ['a', 'a', 'b', 'b'], repetitions=1000)
    assert result['groups'] == 2 and result['n'] == 4
    assert result['mean'] == [.5] and result['interval_95'] == [[0, 1]]
    assert result == grouped_interval([0, 0, 1, 1], ['a', 'a', 'b', 'b'], repetitions=1000)
    assert grouped_interval([1, 0], ['same', 'same'])['interval_95'] is None
    assert grouped_interval([], [])['mean'] is None


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        probability_losses([1], [1.01])


def test_sign_flip_exact_distribution_and_null():
    # Five positive equal effects: only all-positive/all-negative of 32 signs
    # are as extreme, giving two-sided p = 2/32.
    result = paired_sign_flip([1] * 5)
    assert result['exact'] and result['evaluated_resamples'] == 32
    assert result['p_value'] == pytest.approx(2 / 32)
    assert paired_sign_flip([0] * 5)['p_value'] == 1
    assert paired_sign_flip([-1] * 5)['p_value'] == result['p_value']


def test_sign_flip_reproducible_and_validated():
    values = list(range(-10, 10))
    assert paired_sign_flip(values, repetitions=101) == paired_sign_flip(values, repetitions=101)
    with pytest.raises(ValueError):
        paired_sign_flip([1])
    with pytest.raises(ValueError):
        paired_sign_flip([1, float('nan')])
