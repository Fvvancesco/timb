"""Development-only tree/RF probe; does not modify the frozen model or its code."""
import argparse
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256, showdown, save_json
from icon_pokemon.features import TeamFeatures
from icon_pokemon.knowledge import CAPABILITIES, ALL_CAPABILITIES, build_graph, capability_evidence
from icon_pokemon import supervised

GRID = {'decision_tree': (2, 4, 8), 'random_forest': (4, 8, 0)}
SEEDS = (11, 23, 47)


def fit_tree(family, depth, x, y, seed=11):
    if family == 'decision_tree':
        estimator = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=seed)
    elif family == 'random_forest':
        estimator = RandomForestClassifier(n_estimators=200, max_depth=depth or None,
            min_samples_leaf=10, max_features='sqrt', bootstrap=True, n_jobs=4, random_state=seed)
    else:
        raise ValueError(f'Unexpected experimental estimator {family}')
    return estimator.fit(x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb-version', choices=('v1', 'v2'), required=True)
    args = parser.parse_args()
    started = perf_counter()
    path = ROOT / 'data/processed/dataset_v1/catalog.json'
    catalog = json.loads(path.read_text())
    teams = {i: t for i, t in catalog['teams'].items() if t['split'] in ('train', 'validation')}
    inputs = [path, *(ROOT / f'data/processed/dataset_{v}/development_labels.jsonl' for v in ('v2', 'v3', 'v4'))]
    rows = [json.loads(line) for p in inputs[1:] for line in p.read_text().splitlines()]
    examples, outcomes = supervised.load_examples(rows, catalog['teams'])
    allowed = {i for t in teams.values() for i in t['set_ids']}
    records = {r['id']: r for r in catalog['sets'] if r['id'] in allowed}
    payload = {'format': catalog['format'], 'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})}
    mechanics = showdown('export-mechanics.cjs', payload)
    semantics = showdown('export-move-semantics.cjs', payload) if args.kb_version == 'v2' else None
    _, graph = build_graph({**catalog, 'teams': teams, 'sets': list(records.values())},
                           mechanics, move_semantics=semantics)
    evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
    features = TeamFeatures(records, graph, mechanics, evidence,
                            capabilities=ALL_CAPABILITIES if semantics else CAPABILITIES)
    @lru_cache(maxsize=None)
    def vector(a, b):
        return features.pair(teams[a]['set_ids'], teams[b]['set_ids'])
    report = {'purpose': 'Additional development-only experiment requested after the core final evaluation was frozen',
              'kb_version': args.kb_version, 'outcomes': outcomes, 'grid': GRID,
              'tree_min_samples_leaf': 20, 'forest_min_samples_leaf': 10,
              'forest_trees': 200, 'forest_max_features': 'sqrt', 'forest_n_jobs': 4,
              'forest_depth_zero_means': 'unbounded', 'seeds': list(SEEDS),
              'features': list(features.names), 'scikit_learn': version('scikit-learn'),
              'input_hashes': {str(p): sha256(p) for p in inputs},
              'code_hashes': {p: sha256(ROOT / p) for p in ['scripts/probe_tree_models.py',
                  'src/icon_pokemon/supervised.py', 'src/icon_pokemon/features.py',
                  'src/icon_pokemon/knowledge.py', 'src/icon_pokemon/move_semantics.py',
                  'src/icon_pokemon/matchups.py', 'ontology/pokemon.ttl',
                  'ontology/move_semantics.ttl', 'ontology/move_annotations.json',
                  'tools/showdown/export-move-semantics.cjs', 'tools/showdown/export-mechanics.cjs',
                  'tools/showdown/package-lock.json']}, 'runs': {}}
    # Substitute only the estimator factory in this isolated experiment process.
    # The shared splitter, scaler policy (none for trees), tuning and scoring are unchanged.
    original_factory = supervised.fit_model
    try:
        supervised.fit_model = fit_tree
        for seed in SEEDS:
            report['runs'][seed] = supervised.evaluate_nested(examples, set(teams), vector, seed, grid=GRID)
            print(f'Completed {args.kb_version}, seed {seed}', flush=True)
    finally:
        supervised.fit_model = original_factory
    report['elapsed_seconds'] = perf_counter() - started
    output = ROOT / f'reports/trees_v2v3v4_kb_{args.kb_version}.json'
    save_json(output, report)
    print(output)


if __name__ == '__main__':
    main()
