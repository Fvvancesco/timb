from itertools import combinations

import pytest

from icon_pokemon.features import effective_stats
from icon_pokemon.supervised import (evaluate_nested, fit_model, nested_folds, probabilities,
                                     refit_development, fit_diagnostics)


def test_nested_split_keeps_all_replicas_together_and_uses_unseen_teams():
    teams = [f't{i:02}' for i in range(18)]
    battles = [(a, b, 1) for a, b in combinations(teams, 2)]
    battles += [(b, a, 0) for a, b in combinations(teams, 2)]

    for train, test, discarded, inner in nested_folds(battles, teams):
        for fit, held, dropped, total in [(train, test, discarded, len(battles)),
                                         *[(a, b, n, len(train)) for a, b, n in inner]]:
            assert len(fit) + len(held) + dropped == total
            assert {i for a, b, _ in fit for i in (a, b)}.isdisjoint(
                i for a, b, _ in held for i in (a, b))
            for side in (set(fit), set(held)):
                assert all((b, a, 1 - y) in side for a, b, y in side)

    def vector(a, b):
        return int(a[1:]), int(b[1:])

    report = evaluate_nested(battles, teams, vector)
    assert report['status'] == 'ok'
    assert report['aggregate']['neutral']['brier'] == {'mean': 0.25, 'sd': 0}
    assert len(report['folds']) == 3
    model, selection = refit_development(battles, teams, vector)
    assert selection['seed'] == 11
    assert len(selection['validation']) == 6
    assert probabilities(model, [vector('t00', 't01')]).shape == (1,)


def test_reversed_pair_has_complementary_probability():
    model = fit_model('logistic', 0.1, [[4, 1], [5, 1], [1, 5], [2, 4]], [1, 1, 0, 0])
    a, b = probabilities(model, [[5, 1], [1, 5]])
    assert a == pytest.approx(1 - b)


def test_neural_fit_reproducible_scaled_and_pair_symmetric():
    import numpy as np
    rng = np.random.default_rng(13)
    x = rng.normal(size=(128, 4))
    y = (x[:, 0] + x[:, 1] > x[:, 2] + x[:, 3]).astype(int)
    first = fit_model('mlp', 'small_l2_1', x, y, seed=11)
    second = fit_model('mlp', 'small_l2_1', x, y, seed=11)
    assert first.named_steps['standardscaler'].mean_ == pytest.approx(x.mean(axis=0))
    assert not first.named_steps['mlpclassifier'].early_stopping
    p = probabilities(first, x)
    assert p == pytest.approx(probabilities(second, x))
    assert p == pytest.approx(1 - probabilities(first, np.hstack((x[:, 2:], x[:, :2]))))
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert fit_diagnostics(first)['epochs'] > 0
    assert fit_diagnostics(first)['parameter_count'] == 193


def test_effective_stats_use_integer_rounding_and_shedinja_hp():
    record = {'base_stats': {'hp': 1, 'atk': 90, 'def': 45, 'spa': 30, 'spd': 30, 'spe': 40},
              'set': {'nature': 'Adamant', 'level': 50, 'ivs': {'spa': 0}, 'evs': {'atk': 252}}}
    assert effective_stats(record, {'Adamant': {'plus': 'atk', 'minus': 'spa'}}) == {
        'hp': 1, 'atk': 156, 'def': 65, 'spa': 31, 'spd': 50, 'spe': 60}
