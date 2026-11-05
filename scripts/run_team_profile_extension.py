"""Audit set clusters and compare team profile features on development only."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neural_network import MLPClassifier

from run_team_completion_extension import (DATA, OUTPUT, ROOT, MODEL_SEEDS,
    input_matrix, masked_cases, metrics, neural_probabilities, rank_key, score)
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.evaluation import grouped_interval
from icon_pokemon.features import STATS, effective_stats
from icon_pokemon.knowledge import ALL_CAPABILITIES, build_graph, capability_evidence

CATALOG = ROOT / 'data/processed/dataset_v1/catalog.json'
REPORT = OUTPUT / 'profiles'
K_VALUES = range(2, 13)
SEEDS = range(10)


def catalog_features():
    catalog = json.loads(CATALOG.read_text())
    teams = [t for t in catalog['teams'].values() if t['split'] == 'train']
    ids = {i for t in teams for i in t['set_ids']}
    records = sorted((r for r in catalog['sets'] if r['id'] in ids), key=lambda r: r['id'])
    query = {'format': catalog['format'], 'move_ids': sorted({m['id'] for r in records for m in r['moves']})}
    semantics = showdown('export-move-semantics.cjs', query)
    mechanics = showdown('export-mechanics.cjs', query)
    _, graph = build_graph({**catalog, 'teams': teams, 'sets': records}, move_semantics=semantics)
    caps = sorted(ALL_CAPABILITIES)
    names = list(STATS) + caps
    rows = []
    for record in records:
        stats = effective_stats(record, mechanics['natures'])
        evidence = capability_evidence(graph, record)
        rows.append([stats[s] for s in STATS] + [int(c in evidence) for c in caps])
    return records, names, np.asarray(rows, dtype=float)


def member(record):
    return {'set_id': record['id'], 'species': record['species'],
            'item': record['set'].get('item'), 'moves': record['set']['moves']}


def audit():
    records, names, raw = catalog_features()
    stats, caps = np.split(raw, [len(STATS)], axis=1)
    stats = (stats - stats.mean(axis=0)) / np.maximum(stats.std(axis=0), 1e-12)
    caps = (caps - caps.mean(axis=0)) / np.sqrt(caps.var(axis=0).sum())
    # Equal total variance for the two blocks, without inflating rare flags.
    x = np.column_stack((stats / np.sqrt(len(STATS)), caps))
    overview, models = {}, {}
    for k in K_VALUES:
        fits = [KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x) for seed in SEEDS]
        aris = [adjusted_rand_score(a.labels_, b.labels_) for a, b in combinations(fits, 2)]
        best = min(fits, key=lambda fit: fit.inertia_)
        overview[k] = {'inertia': float(best.inertia_), 'ari_mean': float(np.mean(aris)),
                       'ari_min': float(min(aris)), 'silhouette': float(silhouette_score(x, best.labels_)),
                       'sizes': np.bincount(best.labels_, minlength=k).tolist()}
        models[k] = best
    inertia = np.array([overview[k]['inertia'] for k in K_VALUES])
    curve = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    distance = np.linspace(1, 0, len(curve)) - curve
    index = int(np.argmax(distance))
    elbow = K_VALUES[index] if distance[index] >= .1 else None
    candidates = [elbow-1, elbow, elbow+1] if elbow else [3, 4, 6, 8]
    profiles, sheets = {}, {}
    for k in candidates:
        model = models[k]
        d = model.transform(x)
        clusters = []
        for label in range(k):
            members = np.flatnonzero(model.labels_ == label)
            near = sorted(members, key=lambda i: d[i, label])[:3]
            border = sorted(members, key=lambda i: np.partition(d[i], 1)[1] - d[i, label])[:2]
            clusters.append({'cluster': label, 'sets': len(members),
                             'species': len({records[i]['base_species'] for i in members}),
                             'means': dict(zip(names, raw[members].mean(axis=0).tolist(), strict=True)),
                             'near_center': [member(records[i]) for i in near],
                             'boundary': [member(records[i]) for i in border]})
        profiles[k] = np.eye(k)[model.labels_]
        sheets[k] = clusters
    by_species = {}
    for name in sorted({r['base_species'] for r in records}):
        mask = [r['base_species'] == name for r in records]
        by_species[name] = {'sets': sum(mask), 'kb': raw[mask, len(STATS):].mean(axis=0).tolist(),
                            **{f'k{k}': profiles[k][mask].mean(axis=0).tolist() for k in candidates}}
    report = {'scope': 'Train catalog only', 'sets': len(records), 'species': len(by_species),
              'catalog_sha256': sha256(CATALOG), 'features': names, 'seeds': list(SEEDS),
              'n_init_per_seed': 10, 'elbow': elbow, 'elbow_gap': float(distance[index]),
              'elbow_rule': 'Largest vertical gap to the normalized endpoint chord; fallback below 0.1',
              'candidates': candidates, 'global_means': dict(zip(names, raw.mean(axis=0).tolist(), strict=True)),
              'curve': overview, 'sheets': sheets, 'profiles': by_species}
    save_json(REPORT / 'audit.json', report)
    print(json.dumps({k: report[k] for k in ('sets','species','elbow','elbow_gap','candidates','curve')}, indent=2), flush=True)


def profile_matrix(visible, vocabulary, profiles, key):
    width = len(next(iter(profiles.values()))[key])
    lookup = np.zeros((len(vocabulary)+1, width+1), dtype=np.float32)
    for i, name in enumerate(vocabulary):
        if name in profiles:
            lookup[i] = [*profiles[name][key], 1]
    # -1 indexes the last all-zero row. The final feature measures catalog availability.
    block = lookup[visible].mean(axis=1)
    return hstack((input_matrix(visible, len(vocabulary)), csr_matrix(block)), format='csr')


def fit_variant(name, x, target, vx, vy, visible, size, dimension, epochs):
    probabilities = []
    for seed in MODEL_SEEDS:
        model = MLPClassifier(hidden_layer_sizes=(dimension,), activation='relu', solver='adam',
                              alpha=.01, batch_size=256, learning_rate_init=.005, random_state=seed)
        for _ in range(epochs):
            model.partial_fit(x, target, classes=np.arange(size))
        probabilities.append(neural_probabilities(model, vx, visible))
    _, scores = score(np.mean(probabilities, axis=0), vy)
    print(name, metrics(scores), flush=True)
    return name, scores


def evaluate(workers):
    catalog = json.loads((REPORT / 'audit.json').read_text())
    chosen = json.loads((OUTPUT / 'selection.json').read_text())
    if chosen['splits_sha256'] != sha256(DATA):
        raise ValueError('Composition split differs from the previous benchmark')
    teams = json.loads(DATA.read_text())['splits']
    vocabulary = chosen['candidate_species']
    train, train_y = masked_cases(teams['train'], vocabulary)
    val, val_y = masked_cases(teams['validation'], vocabulary)
    profiles = catalog['profiles']
    matrices = {'identity': (input_matrix(train, len(vocabulary)), input_matrix(val, len(vocabulary)))}
    for key in ['kb', *(f'k{k}' for k in catalog['candidates'])]:
        matrices[key] = (profile_matrix(train, vocabulary, profiles, key),
                         profile_matrix(val, vocabulary, profiles, key))
    dims = chosen['pokemon2vec_selected']
    runs = dict(Parallel(n_jobs=workers)(delayed(fit_variant)(
        name, x, train_y, vx, val_y, val, len(vocabulary), dims['dimension'], dims['epochs'])
        for name, (x, vx) in matrices.items()))
    results = {name: metrics(scores) for name, scores in runs.items()}
    groups = np.repeat(np.arange(len(teams['validation'])), 6).tolist()
    best = min((name for name in results if name.startswith('k') and name != 'kb'),
               key=lambda name: rank_key(results[name]))
    report = {'scope': 'Exploratory validation on the previously used split; test predictions were not computed',
              'train_teams': len(teams['train']), 'validation_teams': len(teams['validation']),
              'validation_masks': len(val_y), 'fit': {k:dims[k] for k in ('dimension','epochs')},
              'model_seeds': list(MODEL_SEEDS), 'validation_member_slots_with_profile': float(np.mean(
                  [member in profiles for team in teams['validation'] for member in team])),
              'metrics': results, 'selected_k': int(best[1:]),
              'paired_top1_top5_vs_identity': {name: grouped_interval(scores[:,:2]-runs['identity'][:,:2], groups)
                  for name, scores in runs.items() if name != 'identity'},
              'paired_selected_cluster_vs_kb': grouped_interval(runs[best][:,:2]-runs['kb'][:,:2], groups),
              'hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                         (DATA, REPORT/'audit.json', OUTPUT/'selection.json', Path(__file__))}}
    save_json(REPORT / 'validation.json', report)
    print(json.dumps({'selected_k':report['selected_k'],'metrics':results}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'evaluate'))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.action == 'audit':
        audit()
    else:
        evaluate(args.workers)
