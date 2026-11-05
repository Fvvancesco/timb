"""Estimate the paired coverage effect after verifying all frozen jobs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from export_simulated_labels import export_batch, sha256
from run_coverage_confirmation import METHODS, OUT, dump


def read_batch(batch, jobs, digest, catalog_hash):
    summary = json.loads((batch / 'batch-summary.json').read_text())
    unverified = [r for r in summary['results'] if r['status'] == 'completed' and not r['replay_match']]
    if not unverified:
        yield from export_batch(batch, jobs, digest, catalog_hash)
        return
    # Keep the original record. One v2 public log differs from the replay by
    # a single annotation and failed the same exact-match check as all others.
    job_id = 'coverage-v2-090-coverage-719-0'
    if [r['job_id'] for r in unverified] != [job_id]:
        raise ValueError(f'Unexpected replay failure in {batch}')
    config = json.loads((batch / 'batch-config.json').read_text())
    spec_path = batch / job_id / 'job.json'
    spec = json.loads(spec_path.read_text())
    if (sha256(spec_path) != config['jobs'][job_id] or spec['manifest_job'] != jobs[job_id]
            or spec['manifest_sha256'] != digest or spec['catalog_sha256'] != catalog_hash):
        raise ValueError('Excluded replay does not match frozen input')
    replay = json.loads((batch / job_id / 'replay-check.json').read_text())
    if replay['match'] or replay['seed'] != spec['engine_seed']:
        raise ValueError('Unexpected evidence for excluded replay')
    remaining = [r for r in summary['results'] if r['job_id'] != job_id]
    with TemporaryDirectory(prefix='icon-coverage-v2-verify-') as tmp:
        overlay = Path(tmp)
        (overlay / 'batch-config.json').symlink_to((batch / 'batch-config.json').resolve())
        dump(overlay / 'batch-summary.json', {**summary, 'results': remaining})
        for row in remaining:
            identifier = row['job_id']
            (overlay / identifier).symlink_to((batch / identifier).resolve(), target_is_directory=True)
        yield from export_batch(overlay, jobs, digest, catalog_hash)
    yield {**jobs[job_id], 'status': 'replay_verification_failed', 'outcome': None,
           'censor_reason': 'public_log_annotation_mismatch', 'replay_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    root = args.output
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    planned = {j['job_id']: j for j in manifest['jobs']}
    rows = {}
    assignments = [json.loads(p.read_text()) for p in sorted((root / 'assignments').glob('*.json'))]
    digest = sha256(manifest_path)
    assigned = [r for a in assignments for r in a['requests']]
    if (len(set(assigned)) != len(assigned)
            or set(assigned) != {r['id'] for r in manifest['requests']}
            or any(a['manifest_sha256'] != digest for a in assignments)):
        raise ValueError('Incomplete, overlapping or mismatched assignments')
    for assignment in assignments:
        request_ids = set(assignment['requests'])
        jobs = {k: j for k, j in planned.items() if j['request_id'] in request_ids}
        shard = root / assignment['shard']
        batches = [shard / 'batch']
        if (shard / 'resume-batch').exists():
            batches.append(shard / 'resume-batch')
        for batch in batches:
            for row in read_batch(batch, jobs, digest, manifest['source_sha256']['catalog.json']):
                if row['job_id'] in rows:
                    raise ValueError(f'Duplicate job {row["job_id"]}')
                rows[row['job_id']] = row
    if set(rows) != set(planned):
        raise ValueError(f'Missing {len(set(planned) - set(rows))} jobs')
    observations = defaultdict(dict)
    for row in rows.values():
        outcome = row['outcome']
        if outcome is None:
            continue
        score = .5 if outcome == 'draw' else float(outcome == ('team_a' if row['orientation'] == 0 else 'team_b'))
        for method in row['methods']:
            observations[row['request_id']][row['root'], row['orientation'], method] = score
    paired = []
    for request in manifest['requests']:
        slots = observations[request['id']]
        complete = [(slots[root_seed, side, 'coverage'], slots[root_seed, side, 'neutral'])
                    for root_seed in manifest['root_seeds'] for side in (0, 1)
                    if all((root_seed, side, method) in slots for method in METHODS)]
        if complete:
            a, b = np.mean(complete, axis=0).tolist()
            paired.append({'request': request['id'], 'slots': len(complete), 'coverage': a, 'neutral': b,
                           'difference': a-b, 'source_team_id': request['source_team_id']})
    if not paired:
        raise ValueError('No paired requests completed')
    differences = np.array([r['difference'] for r in paired])
    groups = defaultdict(list)
    for request in paired:
        groups[request['source_team_id']].append(request['difference'])
    sizes = np.array([len(v) for v in groups.values()])
    totals = np.array([sum(v) for v in groups.values()])
    rng = np.random.default_rng(11)
    selections = rng.integers(len(sizes), size=(10000, len(sizes)))
    resamples = totals[selections].sum(axis=1) / sizes[selections].sum(axis=1)
    result = {'manifest_sha256': sha256(manifest_path), 'physical_jobs': len(rows),
              'statuses': dict(Counter(r['status'] for r in rows.values())), 'requests': len(manifest['requests']),
              'paired_requests': len(paired), 'source_team_clusters': len(groups), 'identical_solver_outputs': sum(r['same_team'] for r in manifest['requests']),
              'coverage_score': float(np.mean([r['coverage'] for r in paired])),
              'neutral_score': float(np.mean([r['neutral'] for r in paired])),
              'coverage_minus_neutral': float(differences.mean()),
              'bootstrap_interval_95': np.quantile(resamples, [.025, .975]).tolist(),
              'bootstrap_seed': 11, 'bootstrap_samples': 10000,
              'estimand': 'Mean over requests of paired win-score differences. Natural draw = 0.5. Slots with censored outcomes in either arm omitted. Bootstrap by source team; residual dependence from common catalog and pilot.',
              'per_request': paired}
    dump(root / 'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_request'}, indent=2))


if __name__ == '__main__':
    main()
