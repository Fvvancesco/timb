"""Evaluate prediction of one hidden base species from five visible teammates."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import softmax
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256
from icon_pokemon.evaluation import grouped_interval

DATA = ROOT / 'data/processed/team_completion_v1/splits.json'
OUTPUT = ROOT / 'reports/team_completion_v1'
SPLIT_SEED = 307
MODEL_SEEDS = (11, 23, 47)
ALPHAS = (.1, .5, 1., 2., 5., 10.)
DIMENSIONS = (16, 32)
EPOCHS = (5, 10, 20, 35)
METRICS = ('top1', 'top5', 'log_loss_covered')


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Frozen artifact differs: {path}')
    if not path.exists():
        save_json(path, value)


def prepare(inputs, output):
    from run_human_expanded import ingest

    replays, species, counts = ingest(inputs, 1600)
    teams = sorted({tuple(sorted(species[name]['base_species'] for name in side))
                    for row in replays.values() for side in row['sides']})
    valid = [team for team in teams if len(team) == len(set(team)) == 6]
    order = np.random.default_rng(SPLIT_SEED).permutation(len(valid))
    n_train, n_dev = int(.70 * len(order)), int(.85 * len(order))
    parts = {'train': order[:n_train], 'validation': order[n_train:n_dev], 'test': order[n_dev:]}
    manifest = {'source_sha256': {path.name: sha256(path) for path in inputs},
                'source_counts': counts, 'minimum_rating': 1600, 'format': 'gen9ou',
                'policy': 'Reuse the filtered decisive human preview corpus; outcomes are unused. Deduplicate unordered base-species compositions.',
                'split_seed': SPLIT_SEED, 'excluded_duplicate_base_species': len(teams) - len(valid),
                'splits': {name: [list(valid[i]) for i in ids] for name, ids in parts.items()}}
    freeze(output, manifest)
    print({name: len(ids) for name, ids in parts.items()}, flush=True)


def masked_cases(teams, vocabulary):
    visible = []
    targets = []
    indices = {name: i for i, name in enumerate(vocabulary)}
    for team in teams:
        ids = [indices.get(name, -1) for name in team]
        for hidden, target in enumerate(ids):
            visible.append(ids[:hidden] + ids[hidden + 1:])
            targets.append(target)
    return np.asarray(visible), np.asarray(targets)


def visible_coordinates(visible):
    row, slot = np.where(visible >= 0)
    return row, visible[row, slot]


def input_matrix(visible, size):
    row, col = visible_coordinates(visible)
    return csr_matrix((np.full(len(row), .2, dtype=np.float32), (row, col)),
                      shape=(len(visible), size))


def normalize(logits, visible):
    row, col = visible_coordinates(visible)
    logits[row, col] = -np.inf
    return softmax(logits, axis=1)


def bayes(visible, counts, cooccurrence, alpha=None):
    prior = np.log(counts / counts.sum())
    scores = np.broadcast_to(prior, (len(visible), len(counts))).copy()
    if alpha is not None:
        likelihood = np.log((cooccurrence + alpha) / (counts[:, None] + 2 * alpha))
        for slot in visible.T:
            known = slot >= 0
            scores[known] += likelihood[:, slot[known]].T
    return normalize(scores, visible)


def neural_probabilities(model, matrix, visible):
    return normalize(np.log(np.clip(model.predict_proba(matrix), 1e-15, 1)), visible)


def score(probability, targets):
    top = np.argsort(-probability, axis=1, kind='stable')[:, :5]
    matches = top == targets[:, None]
    covered = targets >= 0
    loss = np.full(len(targets), np.nan)
    loss[covered] = -np.log(np.clip(probability[np.flatnonzero(covered), targets[covered]], 1e-15, 1))
    return top, np.column_stack((matches[:, 0], matches.any(axis=1), loss))


def metrics(scores):
    return dict(zip(METRICS, np.nanmean(scores, axis=0).tolist(), strict=True))


def rank_key(result):
    return (-result['top5'], result['log_loss_covered'])


def train_neural(train_x, train_y, val_x, val_visible, val_y, size):
    selection, best = [], None
    for dimension in DIMENSIONS:
        models = [MLPClassifier(hidden_layer_sizes=(dimension,), activation='relu',
                                solver='adam', alpha=.01, batch_size=256,
                                learning_rate_init=.005, random_state=seed) for seed in MODEL_SEEDS]
        for epoch in range(1, max(EPOCHS) + 1):
            for model in models:
                model.partial_fit(train_x, train_y, classes=np.arange(size))
            if epoch not in EPOCHS:
                continue
            prediction = np.mean([neural_probabilities(m, val_x, val_visible) for m in models], axis=0)
            record = {'dimension': dimension, 'epochs': epoch, **metrics(score(prediction, val_y)[1])}
            selection.append(record)
            print('Pokémon2Vec validation', record, flush=True)
            if best is None or rank_key(record) < rank_key(best[0]):
                import copy
                best = (record, copy.deepcopy(models))
    return best, selection


def evaluate(data, output):
    manifest = json.loads(data.read_text())
    teams = manifest['splits']
    groups = {name: {tuple(team) for team in part} for name, part in teams.items()}
    if any(a & b for a, b in combinations(groups.values(), 2)):
        raise ValueError('Duplicate composition across splits')
    if any(len(groups[k]) != len(v) or any(len(t) != 6 or len(set(t)) != 6 for t in v)
           for k, v in teams.items()):
        raise ValueError('Each split must contain unique six-species teams')
    vocabulary = sorted({species for team in teams['train'] for species in team})
    size = len(vocabulary)
    if size < 10 or any(not part for part in teams.values()):
        raise ValueError('Insufficient species or compositions')
    train_visible, train_y = masked_cases(teams['train'], vocabulary)
    train_x = input_matrix(train_visible, size)
    counts = np.bincount(train_y, minlength=size)
    cooccurrence = np.zeros((size, size), dtype=int)
    for slot in train_visible.T:
        np.add.at(cooccurrence, (train_y, slot), 1)
    val_visible, val_y = masked_cases(teams['validation'], vocabulary)
    bayes_selection = [{'alpha': alpha, **metrics(score(bayes(val_visible, counts, cooccurrence, alpha), val_y)[1])}
                       for alpha in ALPHAS]
    best_bayes = min(bayes_selection, key=rank_key)
    (best_neural, models), neural_selection = train_neural(
        train_x, train_y, input_matrix(val_visible, size), val_visible, val_y, size)
    selected = {'splits_sha256': sha256(data), 'code_sha256': sha256(Path(__file__)),
                'primary': 'top5 over all masks; log loss on common covered cases breaks validation ties',
                'frequency_validation': metrics(score(bayes(val_visible, counts, cooccurrence), val_y)[1]),
                'bayes_grid': bayes_selection, 'bayes_selected': best_bayes,
                'pokemon2vec_grid': neural_selection, 'pokemon2vec_selected': best_neural,
                'model_seeds': list(MODEL_SEEDS), 'candidate_species': vocabulary,
                'out_of_vocabulary': 'Ignore unknown visible species for model features. Unknown targets count as top-k misses and are excluded from covered log loss.'}
    freeze(output / 'selection.json', selected)
    test_visible, test_y = masked_cases(teams['test'], vocabulary)
    test_x = input_matrix(test_visible, size)
    predicted = {'frequency': bayes(test_visible, counts, cooccurrence),
                 'naive_bayes': bayes(test_visible, counts, cooccurrence, best_bayes['alpha']),
                 'pokemon2vec': np.mean([neural_probabilities(m, test_x, test_visible) for m in models], axis=0)}
    ranked, scores = {}, {}
    for name, prob in predicted.items():
        ranked[name], scores[name] = score(prob, test_y)
    team_ids = np.repeat(np.arange(len(teams['test'])), 6).tolist()
    covered = test_y >= 0
    comparisons = {}
    for new, reference in [('naive_bayes', 'frequency'), ('pokemon2vec', 'naive_bayes'), ('pokemon2vec', 'frequency')]:
        delta = scores[new] - scores[reference]
        comparisons[f'{new}_minus_{reference}'] = {
            'top1_top5': grouped_interval(delta[:, :2], team_ids),
            'log_loss_covered': grouped_interval(delta[covered, 2], np.asarray(team_ids)[covered].tolist())}
    report = {'team_counts': {k: len(v) for k, v in teams.items()}, 'test_masks': len(test_y),
              'out_of_vocabulary_targets': int((~covered).sum()), 'masks_with_unknown_visible_species': int((test_visible < 0).any(axis=1).sum()),
              'metrics': {name: metrics(value) for name, value in scores.items()}, 'comparisons': comparisons,
              'test_source': 'Grouped holdout of human team compositions, reusing the human-preview source shards.',
              'selection_sha256': sha256(output / 'selection.json')}
    save_json(output / 'report.json', report)
    with (output / 'test_predictions.jsonl').open('w') as file:
        for row, team_id in enumerate(team_ids):
            team = teams['test'][team_id]
            hidden = row % 6
            entry = {'team_id': team_id, 'visible': team[:hidden] + team[hidden + 1:],
                     'target': team[hidden], 'target_in_vocabulary': bool(covered[row])}
            entry['methods'] = {name: {'top5': [[vocabulary[i], float(prob[row, i])] for i in ranked[name][row]],
                                       'target_probability': float(prob[row, test_y[row]]) if covered[row] else 0.}
                                for name, prob in predicted.items()}
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    parser.add_argument('inputs', nargs='*', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == 'prepare':
        if len(args.inputs) != 2:
            parser.error('prepare requires the PokéChamp and Metamon source files')
        prepare(args.inputs, args.data)
    else:
        evaluate(args.data, args.output)


if __name__ == '__main__':
    main()
