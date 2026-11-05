"""Convert development batch evidence into supervised labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256


def read(path):
    return json.loads(path.read_text())


def export_batch(batch, jobs, manifest_hash, catalog_hash, *, preserve_partial=False):
    config = read(batch / 'batch-config.json')
    summary = read(batch / 'batch-summary.json')
    for entry in summary['results']:
        identifier = entry.get('job_id')
        if identifier not in jobs:
            raise ValueError(f'Unexpected job in {batch}: {entry}')
        if entry['status'] == 'error' and preserve_partial:
            continue
        if entry['status'] == 'error':
            raise ValueError(f'Unexpected or failed job in {batch}: {entry}')
        folder = batch / identifier
        spec_path = folder / 'job.json'
        if sha256(spec_path) != config['jobs'][identifier]:
            raise ValueError(f'Specification hash mismatch for {identifier}')
        spec, result = read(spec_path), read(folder / 'result.json')
        job = jobs[identifier]
        if (spec['manifest_sha256'] != manifest_hash or spec['catalog_sha256'] != catalog_hash
                or spec['manifest_job'] != job or spec['job_id'] != identifier
                or result['job_id'] != identifier):
            raise ValueError(f'Evidence does not match manifest for {identifier}')
        status, outcome = entry['status'], None
        if status == 'censored':
            if (not config.get('allow_known_pilot_censor') or result['status'] != 'error'
                    or result['outcome'] is not None or entry.get('replay_match')
                    or entry.get('censor_reason') != 'foul_play_zero_visit_mcts'):
                raise ValueError(f'Invalid censor record for {identifier}')
        elif status in ('completed', 'truncated', 'timeout'):
            replay = read(folder / 'replay-check.json')
            if (status != result['status'] or not entry['replay_match'] or replay['match'] is not True
                    or replay['seed'] != spec['engine_seed']
                    or result['actual_engine_seed'] != spec['engine_seed']
                    or result['python_seed_state_verified'] is not True):
                raise ValueError(f'Unverified replay or seeds for {identifier}')
            if status == 'completed':
                winners = dict(zip((name.casefold() for name in spec['players']), ('team_a', 'team_b')))
                raw = str(result['outcome']).casefold()
                if raw == 'draw':
                    outcome = 'draw'
                elif raw in winners:
                    outcome = winners[raw]
                else:
                    raise ValueError(f'Unknown winner for {identifier}: {raw}')
            elif result['outcome'] is not None:
                raise ValueError(f'Unexpected outcome for {identifier}: {status}')
        else:
            raise ValueError(f'Unexpected status for {identifier}: {status}')
        yield {**job, 'status': status, 'outcome': outcome,
               'censor_reason': entry.get('censor_reason'), 'replay_verified': entry['replay_match'],
               'actual_engine_seed': result['actual_engine_seed'],
               'completed_turns': result['completed_turns']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batches', nargs='+', type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'config/dataset_v2_manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preserve-partial', action='store_true',
                        help='Export verified results from stopped batches while leaving errors pending.')
    args = parser.parse_args()
    manifest = read(args.manifest)
    jobs = {j['job_id']: j for j in manifest['jobs'] if j['split'] in ('train', 'validation')}
    manifest_hash = sha256(args.manifest)
    catalog_hash = manifest['hashes'][manifest['catalog_path']]
    collected = {}
    for batch in args.batches:
        for row in export_batch(batch, jobs, manifest_hash, catalog_hash,
                                preserve_partial=args.preserve_partial):
            identifier = row['job_id']
            if identifier in collected:
                raise ValueError(f'Duplicate job evidence: {identifier}')
            collected[identifier] = row
    output = ''.join(json.dumps(collected[i], sort_keys=True) + '\n' for i in jobs if i in collected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    temp.write_text(output)
    temp.replace(args.output)
    print(json.dumps({'collected': len(collected), 'pending': len(jobs) - len(collected),
                      'statuses': dict(Counter(row['status'] for row in collected.values()))}))


if __name__ == '__main__':
    main()
