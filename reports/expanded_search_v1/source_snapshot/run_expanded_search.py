"""Resumable, remote-only expanded nested CV; never opens final-test outcomes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from random import Random
import socket
import sys
from time import perf_counter

import joblib
from joblib import Parallel, delayed, parallel_config
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256, showdown
from icon_pokemon.features import TeamFeatures
from icon_pokemon.knowledge import ALL_CAPABILITIES, CAPABILITIES, build_graph, capability_evidence
from icon_pokemon.supervised import fold_members, load_examples, metrics
from icon_pokemon.expanded_search import candidates, fit_score, is_restricted, row_indices, swap

CONFIG = ROOT / 'config/expanded_search_v1.json'
OUT = ROOT / 'reports/expanded_search_v1'


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def freeze(path, data):
    if path.exists():
        if read(path) != data:
            raise ValueError(f'Frozen artifact changed: {path}')
    else:
        atomic(path, data)


def stamps(paths):
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def verify(hashes):
    for name, digest in hashes.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen dependency changed: {name}')


def status(out, **values):
    atomic(out / 'status.json', {'updated_utc': datetime.now(timezone.utc).isoformat(),
                               'host': socket.gethostname(), 'pid': os.getpid(), **values})


def prepare(out):
    config = read(CONFIG)
    reference = out / 'local_source_hashes.json'
    verify(read(reference))
    catalog_path = ROOT / 'data/processed/dataset_v1/catalog.json'
    inputs = [CONFIG, catalog_path, reference]
    inputs += [ROOT / f'data/processed/dataset_{v}/development_labels.jsonl' for v in config['datasets']]
    inputs += [ROOT / f'config/dataset_{v}_manifest.json' for v in config['datasets']]
    code = [Path(__file__), ROOT / 'src/icon_pokemon/expanded_search.py']
    code += [ROOT / f'src/icon_pokemon/{name}.py' for name in
             ('adapter', 'features', 'knowledge', 'move_semantics', 'matchups', 'supervised')]
    code += [ROOT / name for name in ('ontology/pokemon.ttl', 'ontology/move_semantics.ttl',
             'ontology/move_annotations.json', 'tools/showdown/export-mechanics.cjs',
             'tools/showdown/export-move-semantics.cjs', 'tools/showdown/package-lock.json')]
    if (out / 'protocol.json').exists():
        protocol = checked(out)
        print('Existing prepared data verified', protocol['counts'], flush=True)
        return
    catalog = read(catalog_path)
    teams = {k: t for k, t in catalog['teams'].items() if t['split'] in ('train', 'validation')}
    if len({tuple(sorted(t['species_composition'])) for t in catalog['teams'].values()}) != len(catalog['teams']):
        raise ValueError('Repeated base-species composition')
    labels, job_ids = [], set()
    for batch in config['datasets']:
        path = ROOT / f'data/processed/dataset_{batch}/development_labels.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        manifest = read(ROOT / f'config/dataset_{batch}_manifest.json')
        expected = {j['job_id']: j for j in manifest['jobs'] if j['split'] in ('train', 'validation')}
        if len(rows) != len(expected) or {r['job_id'] for r in rows} != set(expected):
            raise ValueError('Incomplete development batch')
        for row in rows:
            if row['job_id'] in job_ids or any(row.get(k) != v for k, v in expected[row['job_id']].items()):
                raise ValueError('Repeated or changed job')
            if row['team_a_id'] not in teams or row['team_b_id'] not in teams:
                raise ValueError('Test team in development')
            job_ids.add(row['job_id'])
        labels.extend(rows)
    examples, outcomes = load_examples(labels, catalog['teams'])
    y = np.array([row[2] for row in examples])
    plans = []
    for seed in config['seeds']:
        for outer, held in enumerate(fold_members(set(teams), seed, config['outer_folds'])):
            dev = set(teams) - held
            outer_split = row_indices(examples, dev, held)
            inner = []
            for valid in fold_members(dev, seed+outer+1, config['inner_folds']):
                split = row_indices(examples, dev-valid, valid)
                # Restrict count to outer development, matching the core protocol.
                split['discarded'] = len(outer_split['train']) - len(split['train']) - len(split['validation'])
                inner.append(split)
            for split in [outer_split, *inner]:
                if any(set(y[split[key]]) != {0, 1} for key in ('train', 'validation')):
                    raise ValueError('Insufficient classes in a fold')
            plans.append({'id': f's{seed}-f{outer}', 'seed': seed, 'outer': outer,
                          'train_teams': sorted(dev), 'heldout_teams': sorted(held),
                          'split': outer_split, 'inner': inner})
    allowed = {i for t in teams.values() for i in t['set_ids']}
    records = {r['id']: r for r in catalog['sets'] if r['id'] in allowed}
    payload = {'format': catalog['format'], 'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})}
    mechanics = showdown('export-mechanics.cjs', payload)
    xs, names = {}, {}
    for representation in config['representations']:
        semantics = showdown('export-move-semantics.cjs', payload) if representation == 'v2' else None
        _, graph = build_graph({**catalog, 'teams': teams, 'sets': list(records.values())},
                               mechanics, move_semantics=semantics)
        evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
        features = TeamFeatures(records, graph, mechanics, evidence,
                                capabilities=ALL_CAPABILITIES if semantics else CAPABILITIES)
        vectors = {}
        for a, b, _ in examples:
            if (a, b) not in vectors:
                vectors[a, b] = features.pair(teams[a]['set_ids'], teams[b]['set_ids'])
        xs[representation] = np.array([vectors[a, b] for a, b, _ in examples])
        for a, b in list(vectors)[:20]:
            reverse = features.pair(teams[b]['set_ids'], teams[a]['set_ids'])
            if not np.allclose(swap(np.array([vectors[a, b]]))[0], reverse):
                raise ValueError('Swapping features differs from recomputing reversed pair')
        names[representation] = features.names
        print('Features ready', representation, xs[representation].shape, flush=True)
    cache = out / 'development.joblib'
    temporary = out / 'development.tmp'
    joblib.dump({'x': xs, 'y': y, 'examples': examples, 'plans': plans, 'names': names}, temporary)
    temporary.replace(cache)
    choices = candidates(config)
    n_inner = len(choices)*len(plans)*len(xs)*len(config['augmentation'])*config['inner_folds']
    protocol = {'config': config, 'input_hashes': stamps(inputs), 'code_hashes': stamps(code),
                'cache_sha256': sha256(cache), 'counts': {'records': len(labels), 'decisive': len(y),
                    'outcomes': outcomes, 'teams': len(teams), 'candidates': dict(Counter(c['family'] for c in choices)),
                    'inner_fits': n_inner, 'outer_plans': len(plans)},
                'versions': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'joblib')},
                'host': socket.gethostname(), 'platform': platform.platform(), 'python': platform.python_version(),
                'split_policy': 'Same team-disjoint outer/inner memberships as original v2-v4 study. Exclude crossing matches; never read final outcomes. All learned preprocessing is fold-local.'}
    freeze(out / 'protocol.json', protocol)
    status(out, state='prepared', counts=protocol['counts'])
    print(json.dumps(protocol['counts']), flush=True)


def checked(out):
    protocol = read(out / 'protocol.json')
    verify(protocol['input_hashes'])
    verify(protocol['code_hashes'])
    if protocol['cache_sha256'] != sha256(out / 'development.joblib'):
        raise ValueError('Prepared matrix changed')
    if protocol['versions'] != {name: version(name) for name in protocol['versions']}:
        raise ValueError('Library versions changed')
    return protocol


def inner_task(candidate, x, y, plan, augment, common):
    folds = [fit_score(candidate, x, y, fold['train'], fold['validation'], plan['seed'], augment, common)
             for fold in plan['inner']]
    return {'candidate': candidate, 'inner': folds,
            'mean_validation_log_loss': float(np.mean([f['validation']['log_loss'] for f in folds]))}


def pilot(out, workers):
    protocol = checked(out)
    data = joblib.load(out / 'development.joblib')  # Trusted project artifact.
    config = protocol['config']
    grid = candidates(config)
    # Small/medium/large settings are fixed by grid position, without scoring-based selection.
    selected = [items[i] for family in config['grids']
                for items in [[c for c in grid if c['family'] == family]]
                for i in (0, len(items)//2, len(items)-1)]
    plan = data['plans'][0]
    split = plan['inner'][0]
    jobs = [(c, aug) for c in selected for aug in config['augmentation']]
    start = perf_counter()
    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=workers)(delayed(fit_score)(c, data['x']['v1'], data['y'],
            split['train'], split['validation'], plan['seed'], aug, config['common']) for c, aug in jobs)
    details = [{'candidate': c, 'augmentation': aug, 'fit_seconds': r['fit']['fit_seconds'],
                'epochs': r['fit'].get('epochs'), 'warnings': r['fit']['warnings']}
               for (c, aug), r in zip(jobs, results, strict=True)]
    rough = sum(np.mean([r['fit_seconds'] for r in details if r['candidate']['family'] == family]) * count
                for family, count in protocol['counts']['candidates'].items())
    rough *= len(data['plans'])*len(data['x'])*len(config['augmentation'])*config['inner_folds']/workers
    result = {'protocol_sha256': sha256(out / 'protocol.json'), 'workers': workers,
              'wall_seconds': perf_counter()-start, 'timings': details,
              'rough_inner_wall_seconds': rough,
              'limits': 'Rough timing from 18 fits on one inner fold; heterogeneous grid costs, outer refits, preprocessing and contention can change total time. No validation scores used to alter grid.'}
    atomic(out / 'timing_pilot.json', result)
    print(json.dumps(result, indent=2), flush=True)


def checkpoint(path, protocol_hash, **identity):
    if not path.exists():
        return None
    data = read(path)
    if data.get('protocol_sha256') != protocol_hash or any(data.get(k) != v for k, v in identity.items()):
        raise ValueError(f'Invalid checkpoint: {path}')
    return data


def summarize(out, protocol, outer_results, complete=False):
    aggregate = {}
    for representation in protocol['config']['representations']:
        for aug in protocol['config']['augmentation']:
            for family in protocol['config']['grids']:
                for scope in ('restricted', 'expanded'):
                    selected = [r for r in outer_results if (r['representation'], r['augmentation'], r['family'], r['scope'])
                                == (representation, aug, family, scope)]
                    if not selected:
                        continue
                    label = f'{representation}/{family}/aug{int(aug)}/{scope}'
                    aggregate[label] = {'folds': len(selected), 'validation': {}, 'train': {}}
                    for part in ('validation', 'train'):
                        for name in ('log_loss', 'brier', 'accuracy'):
                            values = [r['result'][part][name] for r in selected]
                            aggregate[label][part][name] = {'mean': float(np.mean(values)),
                                'sd': float(np.std(values, ddof=1)) if len(values)>1 else None}
    report = {'complete': complete, 'protocol_sha256': sha256(out / 'protocol.json'),
              'aggregate': aggregate, 'outer_results': outer_results,
              'limits': ['Exploratory post-core analysis, no final test evaluation.',
                         'Overlapping outer folds are not independent observations; SD is descriptive.',
                         'Restricted grid uses the same new preprocessing and iteration budget as expanded grid.',
                         'No statistical superiority claim from the smallest development mean alone.']}
    atomic(out / 'summary.json', report)


def run(out, workers):
    protocol = checked(out)
    digest = sha256(out / 'protocol.json')
    data = joblib.load(out / 'development.joblib')
    config, grid = protocol['config'], candidates(protocol['config'])
    outer_results = []
    total_tasks = len(grid)*len(data['plans'])*len(config['augmentation'])*len(data['x'])
    done = 0
    start = perf_counter()
    for representation, x in data['x'].items():
        for plan in data['plans']:
            pending, results = [], {}
            stage = out / 'checkpoints' / representation / plan['id']
            for aug in config['augmentation']:
                for i, choice in enumerate(grid):
                    path = stage / f'aug{int(aug)}-c{i:03d}.json'
                    existing = checkpoint(path, digest, candidate=choice, augmentation=aug, plan=plan['id'])
                    if existing:
                        results[aug, i] = existing
                        done += 1
                    else:
                        pending.append((aug, i, choice, path))
            def task(aug, i, choice, path):
                record = inner_task(choice, x, data['y'], plan, aug, config['common'])
                return aug, i, path, {**record, 'protocol_sha256': digest, 'augmentation': aug, 'plan': plan['id']}
            with parallel_config(backend='loky', inner_max_num_threads=1):
                stream = Parallel(n_jobs=workers, return_as='generator_unordered', batch_size=1)(
                    delayed(task)(*job) for job in pending)
                for aug, i, path, record in stream:
                    atomic(path, record)
                    results[aug, i] = record
                    done += 1
                    status(out, state='running', phase='inner_grid', representation=representation,
                           plan=plan['id'], candidate_tasks_done=done, candidate_tasks_total=total_tasks,
                           elapsed_seconds=perf_counter()-start)
                    if done % 20 == 0:
                        print(f'Candidates {done}/{total_tasks}: {representation} {plan["id"]}', flush=True)
            for aug in config['augmentation']:
                for family in config['grids']:
                    for scope in ('restricted', 'expanded'):
                        options = [i for i, c in enumerate(grid) if c['family'] == family
                                   and (scope == 'expanded' or is_restricted(c))]
                        best = min(options, key=lambda i: results[aug, i]['mean_validation_log_loss'])
                        choice = grid[best]
                        path = stage / f'outer-aug{int(aug)}-c{best:03d}.json'
                        identity = {'candidate': choice, 'augmentation': aug, 'plan': plan['id']}
                        final = checkpoint(path, digest, **identity)
                        if final is None:
                            split = plan['split']
                            result = fit_score(choice, x, data['y'], split['train'], split['validation'],
                                               plan['seed'], aug, config['common'], True)
                            final = {'protocol_sha256': digest, **identity, 'result': result}
                            atomic(path, final)
                        curves = []
                        if scope == 'expanded':
                            order = list(plan['train_teams'])
                            Random(plan['seed']+1000+plan['outer']).shuffle(order)
                            for fraction in config['learning_curve_team_fractions']:
                                if fraction == 1:
                                    curves.append({'fraction': fraction, 'train_teams': len(order), 'result': final['result']})
                                    continue
                                subset = set(order[:max(2, int(len(order)*fraction))])
                                split = row_indices(data['examples'], subset, set(plan['heldout_teams']))
                                curve_path = stage / f'curve-aug{int(aug)}-c{best:03d}-q{fraction}.json'
                                curve = checkpoint(curve_path, digest, **identity, fraction=fraction)
                                if curve is None:
                                    result = fit_score(choice, x, data['y'], split['train'], split['validation'],
                                                       plan['seed'], aug, config['common'])
                                    curve = {'protocol_sha256': digest, **identity, 'fraction': fraction,
                                             'train_teams': len(subset), 'result': result}
                                    atomic(curve_path, curve)
                                curves.append(curve)
                        valid_y = data['y'][plan['split']['validation']]
                        rate = float(data['y'][plan['split']['train']].mean())
                        outer_results.append({**identity, 'representation': representation, 'family': family,
                            'scope': scope, 'result': final['result'], 'learning_curve': curves,
                            'inner_selection_log_loss': results[aug, best]['mean_validation_log_loss'],
                            'neutral': metrics(valid_y, np.full(len(valid_y), .5)),
                            'train_rate': metrics(valid_y, np.full(len(valid_y), rate))})
            summarize(out, protocol, outer_results)
            print('Outer fold complete', representation, plan['id'], flush=True)
    summarize(out, protocol, outer_results, complete=True)
    status(out, state='complete', candidate_tasks_done=done, candidate_tasks_total=total_tasks,
           elapsed_seconds=perf_counter()-start, outer_results=len(outer_results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'pilot', 'run'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if socket.gethostname() != 'Francesco-Desktop':
        parser.error('Training and feature preparation are authorized on Francesco-Desktop only')
    if not 1 <= args.workers <= 14:
        parser.error('Use 1 to 14 workers')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.action == 'prepare':
                prepare(args.output)
            elif args.action == 'pilot':
                pilot(args.output, args.workers)
            else:
                run(args.output, args.workers)
        except BaseException as error:
            status(args.output, state='failed', action=args.action, error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
