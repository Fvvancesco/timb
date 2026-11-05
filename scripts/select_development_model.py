"""Select representation/model using development CV, then fit all development rows.

The reported CV scores are selection scores, not an unbiased final evaluation.
"""
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
import argparse
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256, showdown, save_json
from icon_pokemon.features import TeamFeatures
from icon_pokemon.knowledge import CAPABILITIES, ALL_CAPABILITIES, build_graph, capability_evidence
from icon_pokemon.supervised import (MLP_CONFIGS, MLP_COMMON, fold_members, partition, load_examples,
                                    validation_scores, fit_model, fit_diagnostics, xy, fold_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/selector_development_v1.joblib')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/development_model_selection.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing selected model')
    started = perf_counter()
    catalog_path = ROOT / 'data/processed/dataset_v1/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    teams = {i: t for i, t in catalog['teams'].items() if t['split'] in ('train', 'validation')}
    paths, labels, manifests = [catalog_path], [], []
    for dataset in ('v2', 'v3', 'v4'):
        path = ROOT / f'data/processed/dataset_{dataset}/development_labels.jsonl'
        manifest_path = ROOT / f'config/dataset_{dataset}_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if sha256(catalog_path) != manifest['hashes']['data/processed/dataset_v1/catalog.json']:
            raise ValueError('Catalog differs from simulation manifest')
        jobs = {j['job_id']: j for j in manifest['jobs'] if j['split'] in ('train', 'validation')}
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if {r['job_id'] for r in rows} != set(jobs):
            raise ValueError('Missing development jobs')
        if any(any(row.get(k) != v for k, v in jobs[row['job_id']].items()) for row in rows):
            raise ValueError('Job metadata differs')
        labels.extend(rows)
        paths.extend([path, manifest_path])
    examples, outcomes = load_examples(labels, catalog['teams'])
    ids = {i for t in teams.values() for i in t['set_ids']}
    records = {r['id']: r for r in catalog['sets'] if r['id'] in ids}
    payload = {'format': catalog['format'], 'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})}
    mechanics = showdown('export-mechanics.cjs', payload)
    semantics = showdown('export-move-semantics.cjs', payload)
    _, graph = build_graph({**catalog, 'sets': list(records.values()), 'teams': teams},
                           mechanics, move_semantics=semantics)
    evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
    seed = 11
    splits = [partition(examples, set(teams) - valid, valid) for valid in fold_members(teams, seed)]
    if any({r[2] for r in rows} != {0, 1} for fit, valid, _ in splits for rows in (fit, valid)):
        raise ValueError('Unusable selection folds')
    grid = {'logistic': (0.0001, 0.001, 0.01, 0.1, 1.0), 'boosting': (1, 2, 3),
            'mlp': tuple(MLP_CONFIGS)}
    # Explicit constant competitor; candidate sorting only breaks exact score ties.
    candidates = [{'representation': 'v1', 'family': 'neutral', 'parameter': None,
                   'mean_log_loss': float(np.log(2))}]
    extractors, vectors, diagnostics = {}, {}, {}
    for kb, capabilities in [('v1', CAPABILITIES), ('v2', ALL_CAPABILITIES)]:
        features = TeamFeatures(records, graph, mechanics, evidence, capabilities=capabilities)
        def make_vector(extractor):
            @lru_cache(maxsize=None)
            def vector(a, b):
                return extractor.pair(teams[a]['set_ids'], teams[b]['set_ids'])
            return vector
        vector = make_vector(features)
        extractors[kb], vectors[kb] = features, vector
        diagnostics[kb] = []
        scores = validation_scores(splits, vector, seed, grid, diagnostics=diagnostics[kb])
        candidates.extend({'representation': kb, 'family': family, 'parameter': parameter,
                           'mean_log_loss': score} for (family, parameter), score in scores.items())
        print(f'Completed selection scores for {kb}', flush=True)
    chosen = min(candidates, key=lambda row: row['mean_log_loss'])
    features = extractors[chosen['representation']]
    x, y = xy(examples, vectors[chosen['representation']])
    model = (DummyClassifier(strategy='prior').fit(x, y) if chosen['family'] == 'neutral'
             else fit_model(chosen['family'], chosen['parameter'], x, y, seed))
    code_paths = ['scripts/select_development_model.py', 'src/icon_pokemon/supervised.py',
                  'src/icon_pokemon/features.py', 'src/icon_pokemon/knowledge.py',
                  'src/icon_pokemon/move_semantics.py', 'src/icon_pokemon/matchups.py',
                  'ontology/pokemon.ttl', 'ontology/move_semantics.ttl',
                  'ontology/move_annotations.json', 'tools/showdown/export-move-semantics.cjs',
                  'tools/showdown/export-mechanics.cjs', 'tools/showdown/package-lock.json']
    report = {'purpose': 'Development model selection, not final evaluation', 'seed': seed,
              'outcomes': outcomes, 'decisive': len(examples), 'grid': grid,
              'neural_configs': MLP_CONFIGS, 'neural_common': MLP_COMMON,
              'candidates': candidates, 'chosen': chosen, 'folds': [fold_counts(*s) for s in splits],
              'neural_fit_diagnostics': diagnostics, 'final_fit_diagnostics': fit_diagnostics(model),
              'features': list(features.names), 'scikit_learn': version('scikit-learn'),
              'input_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in paths},
              'code_hashes': {p: sha256(ROOT / p) for p in code_paths},
              'elapsed_seconds': perf_counter() - started,
              'limits': 'Selection scores reuse development data and cannot serve as final performance claims. The held-out test remains unavailable.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    joblib.dump({'estimator': model, 'features': features.names, 'selection': chosen,
                 'kb_version': chosen['representation'], 'capabilities': features.capabilities,
                 'scikit_learn': report['scikit_learn'], 'input_hashes': report['input_hashes'],
                 'code_hashes': report['code_hashes'], 'seed': seed}, temporary)
    temporary.replace(args.output)
    report['model_sha256'] = sha256(args.output)
    save_json(args.report, report)
    print(json.dumps(chosen, indent=2))


if __name__ == '__main__':
    main()
