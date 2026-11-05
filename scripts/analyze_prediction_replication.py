"""Verify the fixed replication cohort and evaluate the frozen predictor."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from analyze_matchup_panel import load
from export_simulated_labels import sha256
from icon_pokemon.evaluation import grouped_interval, probability_losses
from run_coverage_confirmation import ROOT, dump

OUTPUT = ROOT / 'data/processed/prediction_replication_v1'


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
