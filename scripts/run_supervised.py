"""Run the development smoke check or team-disjoint nested evaluation."""
import argparse
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.features import TeamFeatures
from icon_pokemon.knowledge import CAPABILITIES, ALL_CAPABILITIES, build_graph, capability_evidence
from icon_pokemon.matchups import build_matchups
from icon_pokemon.selection import candidate_pool, select_candidates
from icon_pokemon.supervised import (GRID, NN_GRID, MLP_CONFIGS, MLP_COMMON,
                                     evaluate_nested, fit_model, fold_counts, fold_members,
                                     load_examples, metrics, partition, probabilities,
                                     refit_development)

V1 = ROOT / 'data/processed/dataset_v1'
SEEDS = (11, 23, 47)


def smoke_check(examples, teams, records, mechanics, graph, evidence, features, vector):
    observed = {i for a, b, _ in examples for i in (a, b)}
    test_ids = fold_members(observed, SEEDS[0])[0]
    train, test, discarded = partition(examples, observed - test_ids, test_ids)
    if len({y for _, _, y in train}) != 2 or not test:
        raise ValueError('Smoke split needs both training classes and at least one evaluation row')
    x, y = [vector(a, b) for a, b, _ in train], [y for _, _, y in train]
    test_x, test_y = [vector(a, b) for a, b, _ in test], [y for _, _, y in test]
    results = {'fold': fold_counts(train, test, discarded), 'metrics': {
        'neutral': metrics(test_y, [0.5] * len(test)),
        'train_rate': metrics(test_y, [sum(y) / len(y)] * len(test)),
    }}
    models = {}
    for family, option in [('logistic', 0.1), ('boosting', 2)]:
        models[family] = fit_model(family, option, x, y)
        results['metrics'][family] = metrics(test_y, probabilities(models[family], test_x))

    fit_ids = {i for a, b, _ in train for i in (a, b)}
    allowed_sets = sorted({i for team in fit_ids for i in teams[team]['set_ids']})
    target_id = min(i for a, b, _ in test for i in (a, b))
    target = teams[target_id]['set_ids']
    matrix = build_matchups(allowed_sets, target, graph, mechanics)
    available = {cap for identifier in allowed_sets for cap in evidence[identifier]}
    original, required = next(
        (teams[team]['set_ids'], cap)
        for team in sorted(fit_ids) for cap in CAPABILITIES
        if cap in available and not any(cap in evidence[i] for i in teams[team]['set_ids'])
    )
    demos = {}
    for name, request in [('completion', {'fixed': original[:2]}),
                          ('repair', {'fixed': original[:2], 'original': original,
                                      'required': [required]})]:
        pool = candidate_pool([records[i] for i in allowed_sets], evidence,
                              format=mechanics['format'], matchups=matrix, limit=8, **request)
        if name == 'repair' and pool['minimum_replacements_proven'] is None:
            raise RuntimeError(f'Repair demo requires a proved replacement minimum: {pool["stop"]}')
        demos[name] = {'request': request, 'pool': pool,
                       'selection': select_candidates(pool['teams'], target, features, models['logistic'])}
    results['candidate_demo'] = {'opponent_team_id': target_id,
                                 'model': 'fixed logistic C=0.1', 'requests': demos}
    return results


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'nested'], default='smoke')
    parser.add_argument('--catalog', type=Path, default=V1 / 'catalog.json')
    parser.add_argument('--labels', nargs='+', type=Path, default=[
        V1 / 'initial_train_labels.jsonl', V1 / 'distributed_train_labels.jsonl'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-output', type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'config/dataset_v2_manifest.json')
    parser.add_argument('--kb-version', choices=['v1', 'v2'], default='v1',
                        help='v2 includes eight extra inferred move capabilities per team.')
    parser.add_argument('--experiment', choices=['baseline', 'neural'], default='baseline',
                        help='Neural is a separate development-only nested MLP comparison.')
    parser.add_argument('--probe-logistic-c', type=float, nargs=3, metavar='C',
                        help='Evaluate three alternative C values on development only.')
    args = parser.parse_args()
    if args.experiment == 'neural' and (args.mode != 'nested' or args.model_output or args.probe_logistic_c):
        parser.error('Neural experiment needs nested mode, no model export and no logistic probe')
    if args.model_output and args.mode != 'nested':
        parser.error('--model-output requires --mode nested')
    if args.probe_logistic_c and (args.mode != 'nested' or args.model_output
                                 or not all(0 < c < float('inf') for c in args.probe_logistic_c)):
        parser.error('--probe-logistic-c needs positive C values and nested mode without model export')
    output = args.output or ROOT / f'reports/supervised_{args.mode}.json'
    if args.model_output and (args.model_output.exists() or args.model_output.resolve() == output.resolve()):
        parser.error('--model-output must be a new path distinct from the report')
    catalog = json.loads(args.catalog.read_text())
    teams = catalog['teams']
    development = {i for i, t in teams.items() if t['split'] in ('train', 'validation')}
    compositions = {tuple(sorted(t['species_composition'])) for t in teams.values()}
    if len(compositions) != len(teams):
        raise ValueError('Catalog teams must have distinct base-species compositions')
    labels = [json.loads(line) for path in args.labels for line in path.read_text().splitlines() if line]
    examples, outcomes = load_examples(labels, teams)
    inputs = [args.catalog, *args.labels]
    if args.model_output:
        manifest = json.loads(args.manifest.read_text())
        if sha256(args.catalog) != manifest['hashes'][manifest['catalog_path']]:
            raise ValueError('Refit catalog differs from the frozen manifest')
        planned = {j['job_id']: j for j in manifest['jobs'] if j['split'] in ('train', 'validation')}
        if {row['job_id'] for row in labels} != set(planned):
            raise ValueError('Operational refit requires evidence for every planned development job')
        if any(any(row.get(key) != value for key, value in planned[row['job_id']].items())
               for row in labels):
            raise ValueError('Development labels do not match the frozen jobs')
        inputs.append(args.manifest)
    development_sets = {i for team in development for i in teams[team]['set_ids']}
    records = {r['id']: r for r in catalog['sets'] if r['id'] in development_sets}
    development_catalog = {**catalog, 'sets': list(records.values()),
                           'teams': {i: teams[i] for i in development}}
    mechanics = showdown('export-mechanics.cjs', {'format': catalog['format'],
        'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})})
    semantics = None
    if args.kb_version == 'v2':
        semantics = showdown('export-move-semantics.cjs', {'format': catalog['format'],
            'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})})
    _, graph = build_graph(development_catalog, mechanics, move_semantics=semantics)
    evidence = {i: capability_evidence(graph, record) for i, record in records.items()}
    features = TeamFeatures(records, graph, mechanics, evidence,
                            capabilities=ALL_CAPABILITIES if semantics else CAPABILITIES)

    @lru_cache(maxsize=None)
    def vector(a, b):
        return features.pair(teams[a]['set_ids'], teams[b]['set_ids'])

    code_paths = ['src/icon_pokemon/features.py', 'src/icon_pokemon/supervised.py',
                  'src/icon_pokemon/knowledge.py', 'src/icon_pokemon/matchups.py',
                  'ontology/pokemon.ttl', 'ontology/move_annotations.json',
                  'tools/showdown/export-mechanics.cjs', 'tools/showdown/package-lock.json',
                  'scripts/run_supervised.py']
    if semantics:
        code_paths += ['ontology/move_semantics.ttl', 'src/icon_pokemon/move_semantics.py',
                       'tools/showdown/export-move-semantics.cjs']
    report = {'mode': args.mode, 'purpose': 'development only', 'outcomes': outcomes,
              'experiment': args.experiment,
              'kb_version': args.kb_version, 'feature_count': len(features.names),
              'capabilities': list(features.capabilities),
              'code_hashes': {p: sha256(ROOT / p) for p in code_paths},
              'features': list(features.names), 'seeds': list(SEEDS),
              'scikit_learn': version('scikit-learn'),
              'input_hashes': {str(path): sha256(path) for path in inputs}}
    if args.mode == 'smoke':
        report.update(smoke_check(examples, teams, records, mechanics, graph, evidence, features, vector))
        report['nested_readiness'] = evaluate_nested(examples, development, vector)
    else:
        grid = (NN_GRID if args.experiment == 'neural' else
                GRID if not args.probe_logistic_c else {'logistic': args.probe_logistic_c})
        if args.experiment == 'neural':
            report['neural_configs'] = MLP_CONFIGS
            report['neural_common'] = MLP_COMMON
        report['grid'] = grid
        report['runs'] = {seed: evaluate_nested(examples, development, vector, seed, grid=grid)
                          for seed in SEEDS}
        if args.model_output and all(run['status'] == 'ok' for run in report['runs'].values()):
            model, selection = refit_development(examples, development, vector, SEEDS[0])
            report['operational_model'] = selection
            args.model_output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.model_output.with_suffix(args.model_output.suffix + '.tmp')
            joblib.dump({'estimator': model, 'selection': selection, 'features': features.names,
                         'kb_version': args.kb_version, 'capabilities': features.capabilities,
                         'code_hashes': report['code_hashes'],
                         'scikit_learn': report['scikit_learn'], 'input_hashes': report['input_hashes'],
                         'development_train_rate': sum(y for _, _, y in examples) / len(examples)}, temp)
            temp.replace(args.model_output)
    report['elapsed_seconds'] = perf_counter() - started
    save_json(output, report)
    print(output)
    if args.model_output and 'operational_model' not in report:
        raise SystemExit('No model exported. Inspect the reported fold counts.')


if __name__ == '__main__':
    main()
