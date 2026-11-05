"""Learn from decisive simulated matches with team-disjoint validation."""
from collections import Counter
from random import Random
from statistics import mean, stdev
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning


GRID = {'logistic': (0.01, 0.1, 1.0), 'boosting': (1, 2, 3)}
MLP_CONFIGS = {
    'small_l2_1': {'hidden_layer_sizes': (32,), 'alpha': 1.0},
    'small_l2_10': {'hidden_layer_sizes': (32,), 'alpha': 10.0},
    'two_layers_l2_10': {'hidden_layer_sizes': (64, 32), 'alpha': 10.0},
}
MLP_COMMON = {'activation': 'relu', 'solver': 'adam', 'batch_size': 64,
              'learning_rate_init': 0.001, 'max_iter': 200, 'tol': 1e-4,
              'n_iter_no_change': 20, 'early_stopping': False}
NN_GRID = {'mlp': tuple(MLP_CONFIGS)}


def load_examples(labels, teams, *, allowed_splits=('train', 'validation')):
    seen, examples, outcomes = set(), [], Counter()
    for row in labels:
        job = row['job_id']
        if job in seen:
            raise ValueError(f'Duplicate battle job {job}')
        seen.add(job)
        a, b = row['team_a_id'], row['team_b_id']
        if (a == b or row['split'] not in allowed_splits
                or any(teams[i]['split'] != row['split'] for i in (a, b))):
            raise ValueError(f'Unexpected team split or pairing in {job}')
        outcome = row['outcome']
        if outcome not in ('team_a', 'team_b', 'draw', None):
            raise ValueError(f'Unrecognized outcome in {job}: {outcome}')
        outcomes[str(outcome)] += 1
        if outcome in ('team_a', 'team_b'):
            if row.get('replay_verified') is not True:
                raise ValueError(f'Decisive result lacks verified replay in {job}')
            examples.append((a, b, int(outcome == 'team_a')))
    return examples, dict(outcomes)


def fold_members(team_ids, seed, k=3):
    team_ids = sorted(set(team_ids))
    if len(team_ids) < k or k < 2:
        raise ValueError('Too few distinct teams for the requested fold count')
    Random(seed).shuffle(team_ids)
    return [set(team_ids[i::k]) for i in range(k)]


def partition(examples, train, test):
    if train & test:
        raise ValueError('Training and evaluation teams must be disjoint')
    train_rows, test_rows, dropped = [], [], 0
    for example in examples:
        a, b, _ = example
        if a in train and b in train:
            train_rows.append(example)
        elif a in test and b in test:
            test_rows.append(example)
        else:
            dropped += 1
    return train_rows, test_rows, dropped


def nested_folds(examples, team_ids, seed=11):
    all_teams = set(team_ids)
    for index, test in enumerate(fold_members(all_teams, seed)):
        development = all_teams - test
        train_rows, test_rows, dropped = partition(examples, development, test)
        inner = [partition(train_rows, development - valid, valid)
                 for valid in fold_members(development, seed + index + 1)]
        yield train_rows, test_rows, dropped, inner


def fold_counts(train, test, dropped):
    return {'train': len(train), 'test': len(test), 'discarded': dropped,
            'train_class_counts': dict(Counter(row[2] for row in train)),
            'test_class_counts': dict(Counter(row[2] for row in test))}


def fit_model(family, parameter, x, y, seed=11):
    if family == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(C=parameter, max_iter=1000))
    elif family == 'boosting':
        model = GradientBoostingClassifier(max_depth=parameter, n_estimators=80,
                                           min_samples_leaf=5, random_state=seed)
    elif family == 'mlp':
        if parameter not in MLP_CONFIGS:
            raise ValueError(f'Unknown MLP configuration {parameter}')
        # No automatic row-wise validation split: team groups must remain disjoint.
        model = make_pipeline(StandardScaler(), MLPClassifier(
            **MLP_COMMON, **MLP_CONFIGS[parameter], random_state=seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y)
        model.icon_fit_warnings_ = [str(w.message) for w in caught]
        return model
    else:
        raise ValueError(f'Unknown model family {family}')
    return model.fit(x, y)


def fit_diagnostics(model):
    if hasattr(model, 'named_steps') and 'mlpclassifier' in model.named_steps:
        estimator = model.named_steps['mlpclassifier']
        return {'epochs': int(estimator.n_iter_), 'training_loss': float(estimator.loss_),
                'warnings': model.icon_fit_warnings_,
                'parameter_count': sum(w.size for w in estimator.coefs_)
                                   + sum(b.size for b in estimator.intercepts_)}
    return None


def probabilities(model, x):
    x = np.atleast_2d(x)
    if x.shape[1] % 2:
        raise ValueError('Paired features must have two equally sized halves')
    a, b = np.hsplit(x, 2)
    flipped = np.hstack((b, a))
    forward = model.predict_proba(x)[:, 1]
    backward = model.predict_proba(flipped)[:, 1]
    return (forward + 1 - backward) / 2


def metrics(y, p):
    return {'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'brier': float(brier_score_loss(y, p)),
            'accuracy': float(accuracy_score(y, np.asarray(p) >= 0.5))}


def xy(rows, vector):
    return np.asarray([vector(a, b) for a, b, _ in rows]), np.asarray([y for _, _, y in rows])


def validation_scores(splits, vector, seed, grid, *, diagnostics=None):
    matrices = [(xy(fit, vector), xy(valid, vector)) for fit, valid, _ in splits]
    scores = {}
    for family, options in grid.items():
        for option in options:
            losses = []
            for fold, ((fit_x, fit_y), (valid_x, valid_y)) in enumerate(matrices):
                model = fit_model(family, option, fit_x, fit_y, seed)
                losses.append(log_loss(valid_y, probabilities(model, valid_x), labels=[0, 1]))
                detail = fit_diagnostics(model)
                if diagnostics is not None and detail is not None:
                    diagnostics.append({'family': family, 'parameter': option,
                                        'inner_fold': fold, **detail})
            scores[family, option] = mean(losses)
    return scores


def refit_development(examples, team_ids, vector, seed=11):
    """Choose an operational model within development and refit on all of it."""
    development = set(team_ids)
    splits = [partition(examples, development - valid, valid)
              for valid in fold_members(development, seed)]
    if any({row[2] for row in rows} != {0, 1} for fit, valid, _ in splits for rows in (fit, valid)):
        raise ValueError('Operational refit needs both classes in every development fold')
    scores = validation_scores(splits, vector, seed, GRID)
    # Fixed family order breaks exact ties. The final test is unavailable here.
    family, parameter = min(scores, key=scores.__getitem__)
    x, y = xy(examples, vector)
    model = fit_model(family, parameter, x, y, seed)
    return model, {'family': family, 'parameter': parameter, 'seed': seed,
                   'mean_validation_log_loss': scores[family, parameter],
                   'validation': [{'family': f, 'parameter': p, 'log_loss': loss}
                                  for (f, p), loss in scores.items()],
                   'folds': [fold_counts(*split) for split in splits]}


def evaluate_nested(examples, team_ids, vector, seed=11, *, grid=None):
    """Tune each family in inner folds and evaluate every outer fold once."""
    grid = GRID if grid is None else grid
    folds = list(nested_folds(examples, team_ids, seed))
    summaries = [{'outer': fold_counts(train, test, dropped),
                  'inner': [fold_counts(*split) for split in inner]}
                 for train, test, dropped, inner in folds]
    usable = all(
        set(counts[side]) == {0, 1}
        for fold in summaries for counts in [fold['outer'], *fold['inner']]
        for side in ('train_class_counts', 'test_class_counts')
    )
    if not usable:
        return {'status': 'insufficient_classes_in_folds', 'folds': summaries}

    for (train, test, _, inner), report in zip(folds, summaries, strict=True):
        train_x, train_y = xy(train, vector)
        test_x, test_y = xy(test, vector)
        report['metrics'] = {'train_rate': metrics(test_y, np.full(len(test), mean(train_y.tolist()))),
                             'neutral': metrics(test_y, np.full(len(test), 0.5))}
        diagnostics = []
        scores = validation_scores(inner, vector, seed, grid, diagnostics=diagnostics)
        if diagnostics:
            report['inner_fit_diagnostics'] = diagnostics
        for family, options in grid.items():
            losses = [scores[family, option] for option in options]
            parameter = min(zip(losses, options))[1]
            model = fit_model(family, parameter, train_x, train_y, seed)
            report['metrics'][family] = {**metrics(test_y, probabilities(model, test_x)),
                                         'parameter': parameter, 'inner_log_losses': losses}
            detail = fit_diagnostics(model)
            if detail is not None:
                report['metrics'][family]['fit_diagnostics'] = detail
    aggregate = {}
    for method in summaries[0]['metrics']:
        aggregate[method] = {}
        for metric in ('log_loss', 'brier', 'accuracy'):
            scores = [report['metrics'][method][metric] for report in summaries]
            aggregate[method][metric] = {'mean': mean(scores), 'sd': stdev(scores)}
    return {'status': 'ok', 'folds': summaries, 'aggregate': aggregate}
