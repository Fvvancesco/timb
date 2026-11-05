"""Freeze development instances or run budgeted solver comparisons with live JSONL."""
import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter
import platform
import atexit
import fcntl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256, showdown, save_json
from icon_pokemon.knowledge import ALL_CAPABILITIES, build_graph, capability_evidence
from icon_pokemon.matchups import build_matchups
from icon_pokemon.solver_benchmark import benchmark_run, success_curve
from icon_pokemon.validation import TeamValidator

MANIFEST = ROOT / 'config/solver_benchmark_v1.json'
CATALOG = ROOT / 'data/processed/dataset_v1/catalog.json'


def prepare():
    full = json.loads(CATALOG.read_text())
    teams = {i: t for i, t in full['teams'].items() if t['split'] == 'train'}
    allowed = {i for t in teams.values() for i in t['set_ids']}
    records = {r['id']: r for r in full['sets'] if r['id'] in allowed}
    payload = {'format': full['format'], 'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})}
    mechanics = showdown('export-mechanics.cjs', payload)
    semantics = showdown('export-move-semantics.cjs', payload)
    _, graph = build_graph({**full, 'teams': teams, 'sets': list(records.values())},
                           mechanics, move_semantics=semantics)
    evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
    return full['format'], teams, records, graph, mechanics, evidence


def plan(teams, records, evidence):
    rng = Random(20261104)
    roster = sorted(teams)
    rng.shuffle(roster)
    sizes = (24, 64, 160, len(records))
    instances = []
    for index in range(20):
        source, opponent = teams[roster[index]], teams[roster[index + 20]]
        pool_size = sizes[(index // 2) % len(sizes)]
        present = {c for i in source['set_ids'] for c in evidence[i]}
        fixed_species = {records[i]['base_species'] for i in source['set_ids'][:2]}
        candidates = list(ALL_CAPABILITIES)
        candidates = candidates[index % len(candidates):] + candidates[:index % len(candidates)]
        witnesses = {c: sorted(i for i in records if c in evidence[i]
                              and records[i]['base_species'] not in fixed_species)
                     for c in candidates if c not in present}
        required = next(c for c in candidates if witnesses.get(c))
        # Include a structural witness before sampling the rest, without consulting a solver.
        chosen = set(source['set_ids']) | {rng.choice(witnesses[required])}
        chosen.update(rng.sample(sorted(set(records) - chosen), pool_size - len(chosen)))
        # Alternate completion/repair; requesting a missing capability makes repair nontrivial.
        request = {'fixed': source['set_ids'][:2], 'required': [required]}
        if index % 2:
            request['original'] = source['set_ids']
        instances.append({'id': f'instance-{index:02}', 'kind': 'repair' if index % 2 else 'completion',
                          'catalog_ids': sorted(chosen), 'source_team_id': source['id'],
                          'opponent_team_id': opponent['id'], 'opponent_ids': opponent['set_ids'],
                          'request': request})
    return {'purpose': 'Development solver benchmark; no held-out teams or battle outcomes',
            'catalog_sha256': sha256(CATALOG), 'instance_seed': 20261104,
            'pilot_seeds': [11, 23, 37, 53, 71], 'methods': ['random', 'greedy', 'cp_sat'],
            'baseline_policy': {'random': 'Uniform retention count in repair, then random species and set; repeat until budget.',
                                'greedy': 'Repeated construction with random tie breaks; prioritize requested capability coverage, then retention and static quality.'},
            'instances': instances}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--budget', type=float, default=1)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--instances', nargs='+')
    parser.add_argument('--validator-mode', choices=('cold', 'persistent'), default='persistent')
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/solver_pilot')
    args = parser.parse_args()
    lock_path = ROOT / 'data/processed/solver_benchmark.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    run_lock = lock_path.open('a+')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another solver benchmark is running on this host')
    prep_start = perf_counter()
    format, teams, records, graph, mechanics, evidence = prepare()
    candidate_manifest = plan(teams, records, evidence)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != candidate_manifest:
            raise ValueError('Existing frozen solver manifest differs')
    else:
        save_json(MANIFEST, candidate_manifest)
    manifest = candidate_manifest
    if args.plan_only:
        print(MANIFEST)
        return
    seeds = args.seeds or manifest['pilot_seeds']
    if len(set(seeds)) != len(seeds) or args.budget <= 0:
        raise ValueError('Distinct seeds and positive budget required')
    if args.instances and not set(args.instances) <= {i['id'] for i in manifest['instances']}:
        raise ValueError('Unknown instance')
    instances = [i for i in manifest['instances'] if not args.instances or i['id'] in args.instances]
    matrices = {i['id']: build_matchups(i['catalog_ids'], i['opponent_ids'], graph, mechanics) for i in instances}
    validator = None
    validator_startup = perf_counter()
    if args.validator_mode == 'persistent':
        validator = TeamValidator()
        atexit.register(validator.close)
        validator.start()
        # Force lazy format/Dex loading before any timed method gets the first request.
        warm_team = teams[min(teams)]
        warm = validator.validate(format, [records[i]['set'] for i in warm_team['set_ids']], 30)
        if not warm['legal']:
            raise ValueError('Warm-up training team is not legal')
    validator_startup_seconds = perf_counter() - validator_startup
    targets = json.loads(args.targets.read_text())['targets'] if args.targets else {}
    if args.targets and any(i['id'] not in targets or targets[i['id']]['quality'] is None for i in instances):
        raise ValueError('Missing quality target')
    preparation_seconds = perf_counter() - prep_start
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'manifest_sha256': sha256(MANIFEST), 'budget_seconds': args.budget, 'seeds': seeds,
              'instance_ids': [i['id'] for i in instances], 'format': format,
              'validator_mode': args.validator_mode,
              'targets_sha256': sha256(args.targets) if args.targets else None,
              'code_hashes': {p: sha256(ROOT / p) for p in ['src/icon_pokemon/solver_benchmark.py',
                  'src/icon_pokemon/optimization.py', 'src/icon_pokemon/knowledge.py',
                  'src/icon_pokemon/move_semantics.py', 'src/icon_pokemon/matchups.py',
                  'ontology/pokemon.ttl', 'ontology/move_semantics.ttl', 'ontology/move_annotations.json',
                  'tools/showdown/export-move-semantics.cjs', 'tools/showdown/export-mechanics.cjs',
                  'tools/showdown/validate-team.cjs', 'tools/showdown/package-lock.json',
                  'tools/showdown/validate-team-stream.cjs', 'src/icon_pokemon/validation.py',
                  'scripts/run_solver_benchmark.py']},
              'machine': platform.node(), 'platform': platform.platform(),
              'timing_scope': f'Warm KB/matrices; per-run construction, search and validation included ({args.validator_mode} validator). Per-run validation cache. Sequential runs, single CP-SAT thread.'}
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Resume configuration differs')
    save_json(config_path, config)
    runs_path = args.output / 'runs.jsonl'
    runs = [json.loads(line) for line in runs_path.read_text().splitlines()] if runs_path.exists() else []
    done = {(r['instance_id'], r['method'], r['seed']) for r in runs}
    order_rng = Random(177)
    for instance in instances:
        selected = [records[i] for i in instance['catalog_ids']]
        def validate(ids, remaining):
            if validator is not None:
                return validator.validate(format, [records[i]['set'] for i in ids], remaining)
            return showdown('validate-team.cjs', {'format': format, 'sets': [records[i]['set'] for i in ids]}, timeout=remaining)
        for seed in seeds:
            methods = list(manifest['methods'])
            order_rng.shuffle(methods)
            for method in methods:
                if (args.output / 'stop.requested').exists():
                    print('Stopped between runs; completed results preserved.', flush=True)
                    return
                if (instance['id'], method, seed) in done:
                    continue
                result = benchmark_run(method, selected, evidence, instance['request'],
                                       matrices[instance['id']], validate, seed=seed, budget=args.budget)
                row = {'instance_id': instance['id'], 'catalog_size': len(selected), **result}
                with runs_path.open('a') as file:
                    file.write(json.dumps(row, sort_keys=True) + '\n')
                runs.append(row)
                print(json.dumps({'instance': instance['id'], 'method': method, 'seed': seed,
                                  'status': row['status'], 'first': row['first_success_seconds'],
                                  'quality': row['best']['quality'] if row['best'] else None}), flush=True)
                if row['status'] == 'ERROR':
                    raise RuntimeError(row['error'])
    checkpoints = sorted({0, *(x for x in (0.1, 0.25, 0.5, 1, 5, 10) if x <= args.budget), args.budget})
    summaries = {i['id']: {method: success_curve(
        [r for r in runs if r['instance_id'] == i['id'] and r['method'] == method], checkpoints)
        for method in manifest['methods']} for i in instances}
    save_json(args.output / 'summary.json', {'preparation_seconds': preparation_seconds,
        'validator_startup_seconds': validator_startup_seconds,
        'runs': len(runs), 'statuses': dict(Counter(r['status'] for r in runs)),
        'curves_first_valid': summaries,
        'curves_quality': {i['id']: {method: success_curve(
            [r for r in runs if r['instance_id'] == i['id'] and r['method'] == method],
            checkpoints, target=targets[i['id']]['quality']) for method in manifest['methods']}
            for i in instances} if targets else {},
        'scope': config['timing_scope']})


if __name__ == '__main__':
    main()
