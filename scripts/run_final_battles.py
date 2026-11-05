"""Run only the already frozen final jobs, then export verified outcomes."""
import argparse
import fcntl
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools/pilots'))
from icon_pokemon.adapter import sha256, save_json
from result_helpers import showdown_seed
from export_simulated_labels import export_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'data/processed/final_v1')
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args()
    lock = (args.folder / 'collector.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Final evaluation is already running')
    path = args.folder / 'manifest.json'
    manifest = json.loads(path.read_text())
    for relative, expected in {**manifest['hashes'], **manifest['code_hashes']}.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Frozen evaluation input changed: {relative}')
    if sha256(ROOT / manifest['model_path']) != manifest['model_sha256']:
        raise ValueError('Frozen model changed')
    catalog_path = ROOT / manifest['catalog_path']
    catalog = json.loads(catalog_path.read_text())
    if any(j['split'] not in ('test', 'selector_test') for j in manifest['jobs']):
        raise ValueError('Final manifest contains nonfinal jobs')
    if not 1 <= args.workers <= 14:
        raise ValueError('Worker count must be between 1 and 14')
    if not os.environ.get('ICON_PILOT_DIR'):
        raise ValueError('ICON_PILOT_DIR required')
    specs_dir = args.folder / 'specs'
    specs_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    manifest_hash, catalog_hash = sha256(path), sha256(catalog_path)
    for job in manifest['jobs']:
        teams = [catalog['teams'][job[f'showdown_p{side}_team_id']] for side in (1, 2)]
        key = hashlib.sha256(job['job_id'].encode()).hexdigest()[:10]
        files = []
        for side, team in enumerate(teams):
            team_file = specs_dir / f'{job["job_id"]}-team{side}.txt'
            text = team['text'].strip() + '\n'
            if team_file.exists() and team_file.read_text() != text:
                raise ValueError('Existing final team input differs')
            team_file.write_text(text)
            files.append(str(team_file.resolve()))
        spec = {'job_id': job['job_id'], 'format': manifest['format'],
            'players': ['IF' + key + side for side in ('a', 'b')], 'team_files': files,
            'packed_teams': [t['packed'] for t in teams], 'expected_teams': [t['sets'] for t in teams],
            'engine_seed': showdown_seed(job['engine_seed']), 'engine_seed_words': job['engine_seed'],
            'python_seeds': [job['pilot_python_seeds'][f'p{side}'] for side in (1, 2)],
            'max_turns': manifest['max_turns'], 'wall_limit_seconds': manifest['wall_limit_seconds'],
            'manifest_sha256': manifest_hash, 'catalog_sha256': catalog_hash,
            'team_text_sha256': [hashlib.sha256((t['text'].strip() + '\n').encode()).hexdigest() for t in teams],
            'manifest_job': job}
        output = specs_dir / (job['job_id'] + '.json')
        if output.exists() and json.loads(output.read_text()) != spec:
            raise ValueError('Existing final specification differs')
        save_json(output, spec)
        specs.append(str(output.resolve()))
    marker = args.folder / 'evaluation_started.json'
    execution = {'manifest_sha256': manifest_hash, 'model_sha256': manifest['model_sha256'],
                 'runner_sha256': sha256(Path(__file__)), 'workers': args.workers}
    if marker.exists():
        previous = json.loads(marker.read_text())
        if any(previous[k] != v for k, v in execution.items()):
            raise ValueError('Final execution configuration changed')
    else:
        save_json(marker, {**execution, 'started_utc': datetime.now(timezone.utc).isoformat(),
                           'rule': 'No model, representation or selector tuning after this point.'})
    batch = args.folder / 'remote-batch'
    result = subprocess.run([sys.executable, str(ROOT / 'tools/pilots/run_batch.py'),
        '--allow-known-pilot-censor', '--workers', str(args.workers), '--output', str(batch),
        '--stop-file', str(args.folder / 'stop.requested'), *specs], cwd=ROOT)
    if (batch / 'batch-summary.json').exists():
        jobs = {j['job_id']: j for j in manifest['jobs']}
        rows = list(export_batch(batch, jobs, manifest_hash, catalog_hash, preserve_partial=True))
        output = args.folder / 'labels.jsonl'
        temporary = output.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
        temporary.replace(output)
        print(json.dumps({'recorded': len(rows), 'expected': len(jobs),
                          'statuses': dict(Counter(r['status'] for r in rows))}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
