"""Estimate the paired coverage effect after verifying all frozen jobs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from export_simulated_labels import export_batch, sha256
from run_coverage_ablation import COUNTS, METHODS, OUT, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    root = args.output
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    planned = {j['job_id']: j for j in manifest['jobs']}
    rows = {}
    for host in COUNTS:
        jobs = {k: j for k, j in planned.items() if j['host'] == host}
        for row in export_batch(root / host / 'batch', jobs, sha256(manifest_path),
                                manifest['source_sha256']['catalog.json']):
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
                    for root_seed in (613, 617) for side in (0, 1)
                    if all((root_seed, side, method) in slots for method in METHODS)]
        if complete:
            a, b = np.mean(complete, axis=0).tolist()
            paired.append({'request': request['id'], 'slots': len(complete), 'coverage': a, 'neutral': b,
                           'difference': a-b})
    if not paired:
        raise ValueError('No paired requests completed')
    differences = np.array([r['difference'] for r in paired])
    rng = np.random.default_rng(11)
    resamples = differences[rng.integers(len(differences), size=(10000, len(differences)))].mean(axis=1)
    result = {'manifest_sha256': sha256(manifest_path), 'physical_jobs': len(rows),
              'statuses': dict(Counter(r['status'] for r in rows.values())), 'requests': len(manifest['requests']),
              'paired_requests': len(paired), 'identical_solver_outputs': sum(r['same_team'] for r in manifest['requests']),
              'coverage_score': float(np.mean([r['coverage'] for r in paired])),
              'neutral_score': float(np.mean([r['neutral'] for r in paired])),
              'coverage_minus_neutral': float(differences.mean()),
              'bootstrap_interval_95': np.quantile(resamples, [.025, .975]).tolist(),
              'bootstrap_seed': 11, 'bootstrap_samples': 10000,
              'estimand': 'Mean over requests of paired win-score differences. Natural draw = 0.5. Slots with censored outcomes in either arm omitted. Bootstrap over request; residual dependence from common catalog and pilot.',
              'per_request': paired}
    dump(root / 'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_request'}, indent=2))


if __name__ == '__main__':
    main()
