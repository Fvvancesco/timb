"""Descriptive final metrics with resampling at the experimental-unit level."""
import numpy as np


def paired_sign_flip(differences, *, seed=11, repetitions=99999):
    """Exploratory test under independent, sign-exchangeable request blocks.

    One entry per request, already averaged over common decisive trials.
    This tests a symmetry/exchangeability null, not just equality of means.
    """
    from scipy.stats import permutation_test
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite paired request differences required')
    if repetitions < 1:
        raise ValueError('Positive resampling count required')
    result = permutation_test((values,), np.mean, permutation_type='samples',
                              vectorized=True, alternative='two-sided',
                              n_resamples=repetitions, batch=1000, rng=seed)
    return {'requests': len(values), 'mean_difference': float(result.statistic),
            'p_value': float(result.pvalue), 'seed': seed,
            'requested_resamples': repetitions,
            'evaluated_resamples': len(result.null_distribution),
            'exact': repetitions >= 2 ** len(values)}


def probability_losses(y, probabilities):
    y, p = np.asarray(y, dtype=float), np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError('Binary observations and matching probability vectors required')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return np.column_stack((-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
                            (p - y) ** 2, ((p >= .5) == y).astype(float)))


def grouped_interval(values, groups, *, seed=11, repetitions=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError('Finite values and one group per row required')
    if not len(values):
        return {'n': 0, 'groups': 0, 'mean': None, 'interval_95': None}
    identifiers = sorted(set(groups))
    result = {'n': len(values), 'groups': len(identifiers), 'mean': values.mean(axis=0).tolist(),
              'seed': seed, 'repetitions': repetitions}
    if len(identifiers) < 2:
        return {**result, 'interval_95': None}
    indices = {g: i for i, g in enumerate(identifiers)}
    sums = np.zeros((len(identifiers), values.shape[1]))
    counts = np.zeros(len(identifiers))
    for row, group in zip(values, groups, strict=True):
        sums[indices[group]] += row
        counts[indices[group]] += 1
    weights = np.random.default_rng(seed).multinomial(len(identifiers),
        np.full(len(identifiers), 1 / len(identifiers)), size=repetitions)
    sampled = (weights @ sums) / (weights @ counts)[:, None]
    return {**result, 'interval_95': np.quantile(sampled, [.025, .975], axis=0).T.tolist()}
