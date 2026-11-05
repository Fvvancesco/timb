"""Post-core development search, isolated from the frozen experiment code."""
from itertools import product
from time import perf_counter
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .supervised import metrics, probabilities


def candidates(protocol):
    result = []
    for family, grid in protocol['grids'].items():
        names = list(grid)
        for values in product(*(grid[k] for k in names)):
            result.append({'family': family, 'params': dict(zip(names, values, strict=True))})
    return result


def is_restricted(candidate):
    """Original search choices, evaluated under the new common fit policy."""
    p = candidate['params']
    if candidate['family'] == 'logistic':
        return p['C'] in (.01, .1, 1.)
    if candidate['family'] == 'boosting':
        return p['n_estimators'] == 80 and p['learning_rate'] == .1 and p['min_samples_leaf'] == 5
    return (p['learning_rate_init'] == .001 and p['batch_size'] == 64
            and ((p['hidden_layer_sizes'] == [32] and p['alpha'] in (1., 10.))
                 or (p['hidden_layer_sizes'] == [64, 32] and p['alpha'] == 10.)))


def swap(x):
    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError('Two equally sized feature blocks required')
    a, b = np.hsplit(x, 2)
    return np.hstack((b, a))


def scale_pairs(scaler, x):
    a, b = np.hsplit(x, 2)
    return np.hstack((scaler.transform(a), scaler.transform(b)))


def training_data(x, y, augment):
    if augment:
        return np.vstack((x, swap(x))), np.concatenate((y, 1-y)), np.full(2*len(y), .5)
    return x, y, np.ones(len(y))


def fit(candidate, x, y, seed, augment, common):
    started = perf_counter()
    family, params = candidate['family'], candidate['params']
    scaler = None
    if family in ('logistic', 'mlp'):
        # Same train-only preprocessing for both augmentation arms. Pool the
        # two positions before fitting, so scaling commutes with swapping.
        a, b = np.hsplit(x, 2)
        scaler = StandardScaler().fit(np.vstack((a, b)))
        x = scale_pairs(scaler, x)
    x, y, weights = training_data(x, y, augment)
    if family == 'logistic':
        model = LogisticRegression(**params, max_iter=common['logistic_max_iter'], random_state=seed)
    elif family == 'boosting':
        model = GradientBoostingClassifier(**params, random_state=seed)
    elif family == 'mlp':
        params = {**params, 'hidden_layer_sizes': tuple(params['hidden_layer_sizes'])}
        model = MLPClassifier(**params, **common['mlp'], random_state=seed)
    else:
        raise ValueError(f'Unknown family: {family}')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x, y, sample_weight=weights)
    diagnostic = {'fit_seconds': perf_counter()-started,
                  'warnings': sorted({f'{w.category.__name__}: {w.message}' for w in caught}),
                  'training_rows': len(y), 'training_weight_sum': float(weights.sum())}
    if family == 'mlp':
        diagnostic.update(epochs=int(model.n_iter_),
                          optimizer_training_loss=float(model.loss_),
                          training_loss_curve=[float(v) for v in model.loss_curve_])
    return model, scaler, diagnostic


def predict(model, scaler, x):
    if scaler is not None:
        x = scale_pairs(scaler, x)
    return probabilities(model, x), model.predict_proba(x)[:, 1]


def fit_score(candidate, x, y, train, valid, seed, augment, common, keep_predictions=False):
    train, valid = np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
    if not len(train) or not len(valid) or len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
        raise ValueError('Both classes required in both partitions')
    if np.intersect1d(train, valid).size:
        raise ValueError('Overlapping training and validation rows')
    model, scaler, detail = fit(candidate, x[train], y[train], seed, augment, common)
    p_train, raw_train = predict(model, scaler, x[train])
    p_valid, raw_valid = predict(model, scaler, x[valid])
    result = {'train': metrics(y[train], p_train), 'validation': metrics(y[valid], p_valid),
              'raw_train': metrics(y[train], raw_train), 'raw_validation': metrics(y[valid], raw_valid),
              'fit': detail}
    if keep_predictions:
        result.update(validation_indices=valid.tolist(), validation_probabilities=p_valid.tolist())
    return result


def row_indices(examples, fit_teams, valid_teams):
    if fit_teams & valid_teams:
        raise ValueError('Overlapping team groups')
    train, valid, discarded = [], [], 0
    for i, (a, b, _) in enumerate(examples):
        if a in fit_teams and b in fit_teams:
            train.append(i)
        elif a in valid_teams and b in valid_teams:
            valid.append(i)
        else:
            discarded += 1
    return {'train': train, 'validation': valid, 'discarded': discarded}
