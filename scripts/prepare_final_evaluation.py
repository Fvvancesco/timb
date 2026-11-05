"""Freeze predictions and selected teams before any final battle outcome exists."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256, showdown, save_json
from icon_pokemon.dataset import stable_id, engine_seed
from icon_pokemon.features import TeamFeatures
from icon_pokemon.knowledge import ALL_CAPABILITIES, build_graph, capability_evidence
from icon_pokemon.matchups import build_matchups
from icon_pokemon.selection import candidate_pool, select_candidates
from icon_pokemon.supervised import probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/selector_development_v1.joblib')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/final_v1')
    args = parser.parse_args()
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        raise ValueError('Final evaluation already frozen; refusing to regenerate')
    bundle = joblib.load(args.model)
    if bundle['scikit_learn'] != version('scikit-learn'):
        raise ValueError('Model runtime version differs')
    # All feature/model code remains exactly the version used to select and fit the model.
    for relative, expected in bundle['code_hashes'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Model code changed: {relative}')
    path = ROOT / 'data/processed/dataset_v1/catalog.json'
    full = json.loads(path.read_text())
    if sha256(path) != bundle['input_hashes'][str(path.relative_to(ROOT))]:
        raise ValueError('Model catalog changed')
    original_manifest_path = ROOT / 'config/dataset_v2_manifest.json'
    original_manifest = json.loads(original_manifest_path.read_text())
    records = {r['id']: r for r in full['sets']}
    train = {i: t for i, t in full['teams'].items() if t['split'] == 'train'}
    test = {i: t for i, t in full['teams'].items() if t['split'] == 'test'}
    allowed = sorted({i for t in train.values() for i in t['set_ids']})
    payload = {'format': full['format'], 'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})}
    mechanics = showdown('export-mechanics.cjs', payload)
    semantics = showdown('export-move-semantics.cjs', payload)
    _, graph = build_graph(full, mechanics, move_semantics=semantics)
    evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
    features = TeamFeatures(records, graph, mechanics, evidence, capabilities=bundle['capabilities'])
    if tuple(features.names) != tuple(bundle['features']):
        raise ValueError('Model feature schema changed')
    test_jobs = [j for j in original_manifest['jobs'] if j['split'] == 'test']
    assert len(test_jobs) == 80
    frozen_predictions = []
    for job in test_jobs:
        a, b = (full['teams'][job[key]]['set_ids'] for key in ('team_a_id', 'team_b_id'))
        prediction = float(probabilities(bundle['estimator'], [features.pair(a, b)])[0])
        frozen_predictions.append({'job_id': job['job_id'], 'p_team_a': prediction})
    roster = sorted(train)
    Random(20261104).shuffle(roster)
    requests, generated, selection_jobs = [], {}, []
    for index, opponent_id in enumerate(sorted(test)):
        request_started = perf_counter()
        original = train[roster[index]]['set_ids']
        fixed = original[:2]
        current = {c for i in original for c in evidence[i]}
        fixed_species = {records[i]['base_species'] for i in fixed}
        caps = list(ALL_CAPABILITIES)
        caps = caps[index % len(caps):] + caps[:index % len(caps)]
        required = next(c for c in caps if c not in current and any(
            c in evidence[i] and records[i]['base_species'] not in fixed_species for i in allowed))
        request = {'fixed': fixed, 'required': [required]}
        if index % 2:
            request['original'] = original
        opponent = test[opponent_id]['set_ids']
        matrix = build_matchups(allowed, opponent, graph, mechanics)
        pool = candidate_pool([records[i] for i in allowed], evidence,
                              format=full['format'], matchups=matrix, limit=20,
                              time_limit=10, **request)
        identifier = f'request-{index:02}'
        row = {'id': identifier, 'kind': 'repair' if index % 2 else 'completion',
               'request': request, 'opponent_id': opponent_id, 'source_team_id': roster[index],
               'pool': pool}
        if not pool['teams']:
            row['failure'] = 'No legal candidate within budget'
            row['request_seconds'] = perf_counter() - request_started
            requests.append(row)
            continue
        selection_started = perf_counter()
        selection = select_candidates(pool['teams'], opponent, features, bundle['estimator'], seed=11)
        row['selection_seconds'] = perf_counter() - selection_started
        row['selection'] = selection
        choices = {}
        for method, choice in selection['choices'].items():
            ids = sorted(choice['ids'])
            team_id = 'candidate-' + stable_id(ids)
            if team_id not in generated:
                validation = showdown('validate-team.cjs', {'format': full['format'],
                    'sets': [records[i]['set'] for i in ids]})
                if not validation['legal']:
                    raise ValueError('Previously validated candidate became illegal')
                generated[team_id] = {'id': team_id, 'split': 'selector_test', 'set_ids': ids,
                    'species_composition': sorted(records[i]['base_species'] for i in ids),
                    'sets': [records[i]['set'] for i in ids], 'text': validation['text'],
                    'packed': validation['packed'], 'source': 'Frozen constructor output'}
            choices.setdefault(team_id, []).append(method)
            choice['team_id'] = team_id
        # Identical choices share actual battles; comparisons keep their paired outcomes.
        for team_id, methods in sorted(choices.items()):
            pair_id = stable_id([identifier, team_id, opponent_id])
            for root in (401, 409):
                for orientation in (0, 1):
                    a, b = (team_id, opponent_id) if orientation == 0 else (opponent_id, team_id)
                    job_id = f'selector_test-{pair_id}-{root}-{orientation}'
                    selection_jobs.append({'job_id': job_id, 'split': 'selector_test',
                        'pair_id': pair_id, 'root_seed': root, 'orientation': orientation,
                        'team_a_id': a, 'team_b_id': b, 'showdown_p1_team_id': a, 'showdown_p2_team_id': b,
                        'engine_seed': engine_seed(identifier, root),
                        'pilot_python_seeds': {side: int(stable_id([identifier, str(root), str(orientation), side])[:8], 16)
                                               for side in ('p1', 'p2')},
                        'request_id': identifier, 'candidate_id': team_id, 'methods': methods})
        row['request_seconds'] = perf_counter() - request_started
        requests.append(row)
        print(json.dumps({'request': identifier, 'pool_size': len(pool['teams']),
                          'distinct_choices': len(choices), 'repair_minimum': pool['minimum_replacements_proven']}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output / 'catalog.json'
    save_json(catalog_path, {**full, 'teams': {**full['teams'], **generated}})
    save_json(args.output / 'predictions.json', frozen_predictions)
    save_json(args.output / 'requests.json', requests)
    code_paths = ['scripts/prepare_final_evaluation.py', 'src/icon_pokemon/selection.py',
                  'src/icon_pokemon/optimization.py', 'tools/showdown/validate-team.cjs']
    frozen = {k: original_manifest[k] for k in ('format', 'showdown_version', 'pilot', 'pilot_commit',
        'search_time_ms', 'search_parallelism', 'search_threads', 'max_turns', 'wall_limit_seconds')}
    frozen.update(purpose='Final prediction and selector evaluation; no tuning after outcomes',
        model_selection=bundle['selection'], model_sha256=sha256(args.model),
        model_path=str(args.model.relative_to(ROOT)), catalog_path=str(catalog_path.relative_to(ROOT)),
        source_test_manifest_sha256=sha256(original_manifest_path),
        hashes={str(p.relative_to(ROOT)): sha256(p) for p in (catalog_path, args.output / 'predictions.json', args.output / 'requests.json')},
        code_hashes={p: sha256(ROOT / p) for p in code_paths},
        jobs=[*test_jobs, *selection_jobs], request_count=len(requests),
        failed_requests=sum('failure' in r for r in requests),
        seed_policy='Prediction uses the already reserved v2 jobs. Selector comparisons share engine seed per request/root across methods and both sides; native timed search remains nondeterministic.',
        uncertainty_policy='Prediction bootstrap by unordered pair; residual shared-team dependence acknowledged. Selector bootstrap by request including all methods, roots and orientations. Report censored/failed requests separately.')
    save_json(manifest_path, frozen)
    print(json.dumps({'prediction_games': len(test_jobs), 'selector_games': len(selection_jobs),
                      'failed_requests': frozen['failed_requests'], 'manifest': str(manifest_path)}, indent=2))


if __name__ == '__main__':
    main()
