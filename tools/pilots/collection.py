"""Start, stop or inspect a detached development-only collector on one host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'tools/pilots'
CATALOG = ROOT / 'data/processed/dataset_v1/catalog.json'


def is_local_pair(job, dataset):
    local_buckets = 1 if dataset == 'v2' else 3
    return int(job['pair_id'][:8], 16) % 10 < local_buckets


def collect(args, state):
    pilot_dir = Path(os.environ['ICON_PILOT_DIR'])
    for i in range(args.workers):
        if not (pilot_dir / f'worker-server-{i}/icon-patch-hashes.json').is_file():
            raise ValueError(f'Missing prepared worker {i} in {pilot_dir}')
    stop = state / 'stop.requested'
    manifest_path = ROOT / f'config/dataset_{args.dataset}_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    paths = sorted(p.parent for p in state.glob('previous/*/batch-summary.json'))
    prior_ids = set()
    for path in paths:
        prior_ids.update(row['job_id'] for row in json.loads((path / 'batch-summary.json').read_text())['results']
                         if row['status'] == 'censored' or (row['status'] != 'error' and row['replay_match']))
    for split in ('train', 'validation'):
        if stop.exists():
            break
        jobs = [j for j in manifest['jobs'] if j['split'] == split]
        selected = [j for j in jobs if is_local_pair(j, args.dataset) == (args.shard == 'local')
                    and j['job_id'] not in prior_ids]
        if not selected:
            continue
        specs = state / f'{split}-specs'
        batch = state / f'{split}-batch'
        subprocess.run([sys.executable, str(TOOLS / 'make_production_specs.py'),
                        '--manifest', str(manifest_path), '--catalog', str(CATALOG),
                        '--split', split, '--limit', str(len(jobs)), '--output', str(specs)],
                       check=True)
        command = [sys.executable, str(TOOLS / 'run_batch.py'), '--allow-known-pilot-censor',
                   '--stop-file', str(stop), '--workers', str(args.workers),
                   '--output', str(batch)]
        command.extend(str(specs / f'{j["job_id"]}.json') for j in selected)
        print(f'{split}: {len(selected)} jobs on {args.shard}', flush=True)
        result = subprocess.run(command)
        if (batch / 'batch-summary.json').is_file():
            paths.append(str(batch))
        if result.returncode:
            break
    if paths:
        subprocess.run([sys.executable, str(ROOT / 'scripts/export_simulated_labels.py'), '--preserve-partial',
                        '--manifest', str(manifest_path), '--output', str(state / 'labels.jsonl'), *paths],
                       check=True)
    return result.returncode if 'result' in locals() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'stop', 'status', '_run'))
    parser.add_argument('--shard', choices=('local', 'remote'), required=True)
    parser.add_argument('--dataset', choices=('v2', 'v3', 'v4', 'v5'), default='v2')
    parser.add_argument('--workers', type=int, choices=range(1, 15))
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    state = ROOT / 'data/processed' / f'dataset_{args.dataset}' / args.shard
    state.mkdir(parents=True, exist_ok=True)
    stop = state / 'stop.requested'
    if args.action == '_run':
        if args.lock_fd is None or args.workers is None:
            parser.error('Internal run needs the inherited lock and workers')
        with os.fdopen(args.lock_fd):
            return collect(args, state)
    with (state / 'collector.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            running = False
        except BlockingIOError:
            running = True
        if args.action == 'status':
            progress = {}
            for split in ('train', 'validation'):
                file = state / f'{split}-batch/batch-summary.json'
                if file.exists():
                    summary = json.loads(file.read_text())
                    progress[split] = {key: summary[key] for key in ('attempted', 'unscheduled', 'in_flight')}
            print(json.dumps({'shard': args.shard, 'running': running,
                              'stop_requested': stop.exists(), 'progress': progress,
                              'log': str(state / 'collector.log')}))
        elif args.action == 'stop':
            # Also cancels a deferred start that has not acquired this lock yet.
            stop.touch()
            print(f'{args.shard}: ' + ('stop requested' if running else 'stop recorded; collector idle'))
        elif running:
            print(f'{args.shard}: already running')
        else:
            if args.workers is None:
                parser.error('start needs --workers')
            if not os.environ.get('ICON_PILOT_DIR'):
                parser.error('start needs ICON_PILOT_DIR')
            guarded = [ROOT / 'data/processed/solver_benchmark.lock',
                       ROOT / 'data/processed/final_v1/collector.lock']
            guarded += [ROOT / f'data/processed/dataset_{v}/{args.shard}/collector.lock'
                        for v in ('v2', 'v3', 'v4', 'v5') if v != args.dataset]
            for path in guarded:
                if path.exists():
                    with path.open('a+') as other:
                        try:
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            raise RuntimeError(f'Another experiment is active: {path}; defer collection until it finishes')
            stop.unlink(missing_ok=True)
            with (state / 'collector.log').open('ab') as log:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '_run', '--shard', args.shard, '--workers', str(args.workers),
                    '--dataset', args.dataset,
                    '--lock-fd', str(lock.fileno())], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lock.fileno(),))
            print(f'{args.shard}: started PID {process.pid}. Log: {state / "collector.log"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
