"""Paired experiment isolating the static coverage objective of the constructor."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/coverage_ablation_v1'
COUNTS = {'workstation': 38, 'laptop': 11, 'casa': 11}
WORKERS = {'workstation': 14, 'laptop': 4, 'casa': 4}
METHODS = ('coverage', 'neutral')


def dump(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def prepare(output):
    sys.path.insert(0, str(ROOT / 'src'))
    from icon_pokemon.adapter import sha256, showdown
    from icon_pokemon.dataset import engine_seed, stable_id
    from icon_pokemon.knowledge import ALL_CAPABILITIES, build_graph, capability_evidence
    from icon_pokemon.matchups import build_matchups, coverage_score
    from icon_pokemon.selection import candidate_pool

    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Manifest already frozen')
    catalog_path = ROOT / 'data/processed/dataset_v1/catalog.json'
    archive_path = ROOT / 'data/raw/teams/modern_replays-gen9ou.tar.gz'
    catalog = json.loads(catalog_path.read_text())
    train = {k: t for k, t in catalog['teams'].items() if t['split'] == 'train'}
    allowed = {i for t in train.values() for i in t['set_ids']}
    old = {tuple(t['species_composition']) for t in catalog['teams'].values()}
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive if m.isfile() and m.name.endswith('.gen9ou_team')]
        sample = set(Random(251105).sample(range(len(members)), min(len(members), 1000)))
        sources = [{'path': m.name, 'text': archive.extractfile(m).read().decode()}
                   for i, m in enumerate(members) if i in sample]
    Random(351105).shuffle(sources)
    opponents, opponent_records = [], {}
    for start in range(0, len(sources), 100):
        new = showdown('export-catalog.cjs', {'format': catalog['format'], 'teams': sources[start:start+100]}, timeout=120)
        indexed = {r['id']: r for r in new['sets']}
        for t in new['teams']:
            composition = tuple(t['species_composition'])
            if (not t['legal'] or composition in old or len(t['set_ids']) != 6
                    or any(len(indexed[i]['moves']) != 4 for i in t['set_ids'])):
                continue
            opponents.append(t)
            old.add(composition)
            opponent_records.update({i: indexed[i] for i in t['set_ids']})
            if len(opponents) == sum(COUNTS.values()):
                break
        if len(opponents) == sum(COUNTS.values()):
            break
    if len(opponents) != sum(COUNTS.values()):
        raise ValueError(f'Only {len(opponents)} distinct new opponents')
    records = {r['id']: r for r in catalog['sets'] if r['id'] in allowed}
    all_records = {**opponent_records, **records}
    all_records = {i: all_records[i] for i in sorted(all_records)}
    moves = sorted({m['id'] for r in all_records.values() for m in r['moves']})
    payload = {'format': catalog['format'], 'move_ids': moves}
    mechanics = showdown('export-mechanics.cjs', payload)
    semantics = showdown('export-move-semantics.cjs', payload)
    _, graph = build_graph({**catalog, 'teams': train, 'sets': list(all_records.values())},
                           mechanics, move_semantics=semantics)
    evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
    seeds = (613, 617)
    roster = Random(902).sample(sorted(train), len(opponents))
    assignments = [host for host, n in COUNTS.items() for _ in range(n)]
    Random(903).shuffle(assignments)
    rows, jobs = [], []

    def validated(ids):
        sets = [all_records[i]['set'] for i in ids]
        result = showdown('validate-team.cjs', {'format': catalog['format'], 'sets': sets})
        if not result['legal']:
            raise ValueError(result['errors'])
        return {'sets': sets, 'text': result['text'].strip() + '\n', 'packed': result['packed']}

    for index, (opponent, source_id, host) in enumerate(zip(opponents, roster, assignments, strict=True)):
        request_id = f'coverage-v1-{index:02}'
        original = train[source_id]['set_ids']
        fixed = original[:2]
        present = {c for i in original for c in evidence[i]}
        capabilities = list(ALL_CAPABILITIES)
        rotated = capabilities[index % len(capabilities):] + capabilities[:index % len(capabilities)]
        required = [next(c for c in rotated if c not in present and any(c in ev for ev in evidence.values()))]
        request = {'fixed': fixed, 'required': required, 'original': original if index % 2 else []}
        matrix = build_matchups(sorted(allowed), opponent['set_ids'], graph, mechanics)
        neutral = {i: {o: {'offensive': False, 'defensive': False} for o in row} for i, row in matrix.items()}
        choices = {}
        for method, objective in zip(METHODS, (matrix, neutral), strict=True):
            pool = candidate_pool(list(records.values()), evidence, format=catalog['format'], matchups=objective,
                                  limit=1, time_limit=10, seed=11, **request)
            if len(pool['teams']) != 1 or pool['teams'][0]['status'] != 'OPTIMAL':
                raise ValueError(f'No certified legal solution for {request_id} {method}')
            team = pool['teams'][0]
            choices[method] = {**team, 'static_score': coverage_score(team['selected_ids'], matrix)}
        if choices['coverage']['replacements'] != choices['neutral']['replacements']:
            raise ValueError(f'Arm replacement counts differ for {request_id}')
        if tuple(choices['coverage']['static_score']) < tuple(choices['neutral']['static_score']):
            raise ValueError(f'Coverage arm scores below neutral for {request_id}')
        same = choices['coverage']['selected_ids'] == choices['neutral']['selected_ids']
        rows.append({'id': request_id, 'host': host, 'opponent': opponent['source'], 'request': request,
                     'choices': choices, 'same_team': same})
        opposition = validated(opponent['set_ids'])
        for method in METHODS[:1] if same else METHODS:
            candidate = validated(choices[method]['selected_ids'])
            for seed in seeds:
                for orientation in (0, 1):
                    teams = [candidate, opposition] if orientation == 0 else [opposition, candidate]
                    key = f'{request_id}-{method}-{seed}-{orientation}'
                    player = stable_id([key])[:10]
                    jobs.append({'job_id': key, 'request_id': request_id, 'host': host,
                                 'methods': list(METHODS) if same else [method], 'root': seed,
                                 'orientation': orientation, 'teams': teams, 'players': ['IC'+player+'a','IC'+player+'b'],
                                 'engine_seed': engine_seed(request_id, seed),
                                 'python_seeds': [int(stable_id([request_id,str(seed),str(orientation),str(side)])[:8],16)
                                                  for side in (0, 1)]})
        print(request_id, host, same, flush=True)
    frozen = {'created_utc': datetime.now(timezone.utc).isoformat(), 'format': catalog['format'],
              'source_sha256': {p.name: sha256(p) for p in (catalog_path, archive_path, Path(__file__))},
              'request_counts': COUNTS, 'workers': WORKERS, 'requests': rows, 'jobs': jobs,
              'protocol': 'One certified legal solution per arm. Identical matchup axis, constraints, repair priority, seed and budget. Neutral coverage coefficients are zero by masking all answers. New distinct opponent compositions outside original catalog; paired by request.'}
    dump(output / 'manifest.json', frozen)
    print('Frozen', len(rows), 'requests;', len(jobs), 'physical battles', dict(Counter(j['host'] for j in jobs)), flush=True)


def run(output, host, pilot_dir, smoke):
    import fcntl
    sys.path.insert(0, str(pilot_dir.resolve()))
    from result_helpers import showdown_seed
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest['source_sha256'][Path(__file__).name]:
        raise ValueError('Runner differs from the frozen manifest')
    work = output / ('smoke-' + host if smoke else host)
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [j for j in manifest['jobs'] if j['host'] == host]
        if smoke:
            jobs = jobs[:4]
        specs = []
        for job in jobs:
            prefix = work / job['job_id']
            files = [prefix.with_suffix(f'.team{side}.txt') for side in (0,1)]
            for path, team in zip(files, job['teams'], strict=True):
                path.write_text(team['text'])
            spec = {'job_id': job['job_id'], 'format': manifest['format'], 'players': job['players'],
                    'team_files': [str(p.resolve()) for p in files],
                    'packed_teams': [t['packed'] for t in job['teams']],
                    'expected_teams': [t['sets'] for t in job['teams']],
                    'engine_seed_words': job['engine_seed'], 'engine_seed': showdown_seed(job['engine_seed']),
                    'python_seeds': job['python_seeds'], 'max_turns': 200, 'wall_limit_seconds': 120,
                    'manifest_sha256': manifest_hash, 'catalog_sha256': manifest['source_sha256']['catalog.json'],
                    'manifest_job': job,
                    'team_text_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]}
            spec_path = prefix.with_suffix('.json')
            dump(spec_path, spec)
            specs.append(str(spec_path))
        subprocess.run([sys.executable, str(pilot_dir / 'run_batch.py'), '--workers', str(4 if smoke else WORKERS[host]),
                        '--allow-known-pilot-censor', '--output', str(work / 'batch'), *specs],
                       env={**os.environ, 'ICON_PILOT_DIR': str(pilot_dir.resolve())}, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','smoke'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--host', choices=COUNTS)
    parser.add_argument('--pilot-dir', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    else:
        if not args.host or not args.pilot_dir:
            parser.error('--host and --pilot-dir are required for execution')
        run(args.output, args.host, args.pilot_dir, args.action == 'smoke')


if __name__ == '__main__':
    main()
