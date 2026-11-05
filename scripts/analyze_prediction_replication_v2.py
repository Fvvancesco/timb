"""Verify the fixed replication cohort and evaluate the frozen predictor."""
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from export_simulated_labels import export_batch, sha256
from icon_pokemon.evaluation import grouped_interval, probability_losses
from run_coverage_confirmation import ROOT, dump

OUTPUT = ROOT / 'data/processed/prediction_replication_v2'


def load(path):
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    digest = sha256(manifest_path)
    for relative, expected in manifest['prior_sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Prior frozen input changed: {relative}')
    assignments = [json.loads(p.read_text()) for p in sorted((path / 'assignments').glob('*.json'))]
    assigned = [pair for assignment in assignments for pair in assignment['requests']]
    if (len(assigned) != len(set(assigned)) or set(assigned) != {r['id'] for r in manifest['requests']}
            or any(a['manifest_sha256'] != digest for a in assignments)):
        raise ValueError('Assignments must cover the frozen cohort exactly')
    jobs = {job['job_id']: job for job in manifest['jobs']}
    rows = {}
    catalog_hash = manifest['source_sha256']['catalog.json']
    for assignment in assignments:
        shard = path / assignment['shard']
        subset = {i: j for i, j in jobs.items() if j['request_id'] in assignment['requests']}
        for batch in [shard / 'batch', *sorted(shard.glob('resume-*'))]:
            for row in read_batch(batch, subset, digest, catalog_hash):
                if row['job_id'] in rows:
                    raise ValueError(f'Duplicate battle: {row["job_id"]}')
                rows[row['job_id']] = row
    if rows.keys() != jobs.keys():
        raise ValueError(f'{len(jobs.keys() - rows.keys())} frozen jobs missing')
    return manifest, digest, rows


def read_batch(batch, jobs, digest, catalog_hash):
    summary = json.loads((batch / 'batch-summary.json').read_text())
    config_path = batch / 'batch-config.json'
    config = json.loads(config_path.read_text())
    verified = []
    for entry in summary['results']:
        if entry['status'] == 'censored' or entry['replay_match']:
            verified.append(entry)
            continue
        job_id = entry['job_id']
        spec_path = batch / job_id / 'job.json'
        spec = json.loads(spec_path.read_text())
        if (sha256(spec_path) != config['jobs'][job_id] or spec['manifest_job'] != jobs[job_id]
                or spec['manifest_sha256'] != digest or spec['catalog_sha256'] != catalog_hash):
            raise ValueError(f'Invalid evidence for excluded job: {job_id}')
        status = 'technical_error' if entry['status'] == 'error' else 'replay_verification_failed'
        yield {**jobs[job_id], 'status': status, 'outcome': None}
    if len(verified) == len(summary['results']):
        yield from export_batch(batch, jobs, digest, catalog_hash)
        return
    # Preserve the original evidence. The common verifier sees only retained records.
    with TemporaryDirectory(prefix='icon-prediction-verify-') as temporary:
        overlay = Path(temporary)
        (overlay / 'batch-config.json').symlink_to(config_path.resolve())
        dump(overlay / 'batch-summary.json', {**summary, 'results': verified})
        for entry in verified:
            job_id = entry['job_id']
            (overlay / job_id).symlink_to((batch / job_id).resolve(), target_is_directory=True)
        yield from export_batch(overlay, jobs, digest, catalog_hash)


def main():
    manifest, digest, rows = load(OUTPUT)
    if sha256(Path(__file__)) != manifest['source_sha256'][Path(__file__).name]:
        raise ValueError('Analysis differs from the frozen protocol')
    usable = [row for row in rows.values() if row['outcome'] in ('team_a', 'team_b')]
    y = np.asarray([row['outcome'] == 'team_a' for row in usable])
    p = np.asarray([row['p_team_a'] for row in usable])
    groups = [row['request_id'] for row in usable]
    identifiers = sorted(set(groups))
    indices = [np.flatnonzero(np.asarray(groups) == identifier) for identifier in identifiers]
    rng = np.random.default_rng(11)
    sampled_auc = []
    for selected in rng.integers(0, len(indices), size=(10000, len(indices))):
        sample = np.concatenate([indices[i] for i in selected])
        labels = y[sample]
        if labels.any() and not labels.all():
            sampled_auc.append(roc_auc_score(labels, p[sample]))
    losses = probability_losses(y, p)
    baseline = probability_losses(y, np.full(len(y), .5))
    summary = {'manifest_sha256': digest, 'pairs_planned': len(manifest['requests']),
               'battles_planned': len(manifest['jobs']), 'battles_decisive': len(usable),
               'statuses': dict(Counter(row['status'] for row in rows.values())),
               'outcomes': dict(Counter(str(row['outcome']) for row in rows.values())),
               'metrics_order': ['log_loss', 'brier', 'accuracy'],
               'model': grouped_interval(losses, groups),
               'constant': grouped_interval(baseline, groups),
               'model_minus_constant': grouped_interval(losses - baseline, groups),
               'auroc': float(roc_auc_score(y, p)),
               'auroc_interval_95': np.quantile(sampled_auc, [.025, .975]).tolist(),
               'auroc_bootstrap_samples': len(sampled_auc),
               'confusion_matrix': {'labels': ['team_b', 'team_a'], 'threshold': .5,
                                    'counts': confusion_matrix(y, p >= .5, labels=[False, True]).tolist()}}
    dump(OUTPUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
