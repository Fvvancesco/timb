"""Evaluate frozen predictions and selectors once all final jobs are accounted for."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256
from icon_pokemon.evaluation import grouped_interval, probability_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'data/processed/final_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/final_evaluation.json')
    args = parser.parse_args()
    manifest_path = args.folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    predictions_path = args.folder / 'predictions.json'
    predictions = {r['job_id']: r['p_team_a'] for r in json.loads(predictions_path.read_text())}
    rows = [json.loads(line) for line in (args.folder / 'labels.jsonl').read_text().splitlines()]
    expected = {j['job_id']: j for j in manifest['jobs']}
    if len(rows) != len(expected) or {r['job_id'] for r in rows} != expected.keys():
        raise ValueError('Final jobs are incomplete or duplicated')
    if any(any(row.get(k) != v for k, v in expected[row['job_id']].items()) for row in rows):
        raise ValueError('Final job metadata differs')
    if any(not r['replay_verified'] for r in rows if r['outcome'] in ('team_a', 'team_b', 'draw')):
        raise ValueError('Unverified decisive outcome')
    if sha256(predictions_path) != manifest['hashes'][str(predictions_path.relative_to(ROOT))]:
        raise ValueError('Frozen predictions changed')
    test_rows = [r for r in rows if r['split'] == 'test']
    usable = [r for r in test_rows if r['outcome'] in ('team_a', 'team_b')]
    y = [int(r['outcome'] == 'team_a') for r in usable]
    probabilities = [predictions[r['job_id']] for r in usable]
    groups = [r['pair_id'] for r in usable]
    loss = probability_losses(y, probabilities)
    baseline = probability_losses(y, np.full(len(y), .5))
    prediction_report = {'total': len(test_rows), 'statuses': dict(Counter(r['status'] for r in test_rows)),
        'outcomes': dict(Counter(str(r['outcome']) for r in test_rows)),
        'metrics_order': ['log_loss', 'brier', 'accuracy'],
        'model': grouped_interval(loss, groups), 'neutral': grouped_interval(baseline, groups),
        'model_minus_neutral': grouped_interval(loss - baseline, groups)}
    methods = ('random', 'symbolic', 'supervised')
    observations = {method: {} for method in methods}
    statuses = {method: Counter() for method in methods}
    draws = Counter()
    for row in rows:
        if row['split'] != 'selector_test':
            continue
        value = None
        if row['outcome'] in ('team_a', 'team_b'):
            winner = row['team_a_id'] if row['outcome'] == 'team_a' else row['team_b_id']
            value = int(winner == row['candidate_id'])
        for method in row['methods']:
            statuses[method][row['status']] += 1
            draws[method] += int(row['outcome'] == 'draw')
            request = observations[method].setdefault(row['request_id'], {})
            request[row['root_seed'], row['orientation']] = value
    selector_report = {}
    for method in methods:
        rates, request_ids, wins, decisive = [], [], 0, 0
        for identifier, trials in sorted(observations[method].items()):
            values = [value for value in trials.values() if value is not None]
            wins += sum(values)
            decisive += len(values)
            if values:
                rates.append(sum(values) / len(values))
                request_ids.append(identifier)
        selector_report[method] = {'statuses': dict(statuses[method]), 'wins': wins,
                                  'draws': draws[method],
                                  'decisive': decisive,
                                  'mean_request_win_rate': grouped_interval(rates, request_ids)}
    paired = {}
    for a, b in [('supervised', 'symbolic'), ('supervised', 'random'), ('symbolic', 'random')]:
        differences, identifiers, trial_count = [], [], 0
        for identifier in sorted(set(observations[a]) & set(observations[b])):
            ra, rb = observations[a][identifier], observations[b][identifier]
            common = [key for key in ra.keys() & rb.keys() if ra[key] is not None and rb[key] is not None]
            if common:
                differences.append(sum(ra[key] - rb[key] for key in common) / len(common))
                identifiers.append(identifier)
                trial_count += len(common)
        paired[a + '_minus_' + b] = {'paired_trials': trial_count,
                                    **grouped_interval(differences, identifiers)}
    report = {'scope': 'Final evaluation of the frozen core system, conditional on Foul Play and fixed format',
              'model_selection': manifest['model_selection'], 'prediction': prediction_report,
              'selectors': selector_report, 'paired_selector_differences': paired,
              'planned_requests': manifest['request_count'], 'failed_constructions': manifest['failed_requests'],
              'physical_battles': len(rows),
              'hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                         (manifest_path, predictions_path, args.folder / 'labels.jsonl')},
              'limits': ['Percentile bootstrap, 10000 repetitions. Prediction resamples unordered pairs; residual dependence from shared teams remains.',
                         'Selector comparisons resample requests, preserving both orientations and roots. Identical selected teams share physical games.',
                         'Primary win rates condition on decisive outcomes. Draws and censored games are excluded and counted separately.',
                         'Paired selector differences use only trials decisive for both methods and weight available requests equally.',
                         'Only 20 reserved opponent teams; no claim of universal team strength or generalization to other pilots/formats.']}
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
