"""Exploratory paired tests and censoring sensitivity after the frozen evaluation.

Does not refit models, select candidates or launch any battle.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256
from icon_pokemon.evaluation import paired_sign_flip


def analyze(rows, evaluation):
    methods = ('random', 'symbolic', 'supervised')
    observations = {method: {} for method in methods}
    for row in rows:
        if row['split'] != 'selector_test':
            continue
        value = None
        if row['outcome'] in ('team_a', 'team_b'):
            winner = row['team_a_id'] if row['outcome'] == 'team_a' else row['team_b_id']
            value = int(winner == row['candidate_id'])
        elif row['outcome'] == 'draw':
            raise ValueError('Draw policy must be specified before applying this binary sensitivity analysis')
        for method in row['methods']:
            trials = observations[method].setdefault(row['request_id'], {})
            slot = row['root_seed'], row['orientation']
            if slot in trials:
                raise ValueError('Duplicate trial for method and request')
            trials[slot] = value
    contrasts = (('supervised', 'symbolic'), ('supervised', 'random'), ('symbolic', 'random'))
    results = {}
    for a, b in contrasts:
        differences, ids, count = [], [], 0
        for request_id in sorted(observations[a].keys() & observations[b].keys()):
            ra, rb = observations[a][request_id], observations[b][request_id]
            common = sorted(k for k in ra.keys() & rb.keys() if ra[k] is not None and rb[k] is not None)
            if common:
                differences.append(sum(ra[k] - rb[k] for k in common) / len(common))
                ids.append(request_id)
                count += len(common)
        name = a + '_minus_' + b
        result = paired_sign_flip(differences)
        expected = evaluation['paired_selector_differences'][name]
        if abs(result['mean_difference'] - expected['mean'][0]) > 1e-12 or count != expected['paired_trials']:
            raise ValueError('Paired estimand disagrees with frozen evaluation report')
        results[name] = {**result, 'paired_trials': count,
                         'p_bonferroni_three_contrasts': min(1., len(contrasts) * result['p_value']),
                         'request_ids': ids, 'request_differences': differences}
    bounds = {}
    for method, requests in observations.items():
        limits = []
        for trials in requests.values():
            wins = sum(x for x in trials.values() if x is not None)
            missing = sum(x is None for x in trials.values())
            limits.append((wins / len(trials), (wins + missing) / len(trials)))
        bounds[method] = {'requests': len(limits),
                         'observed_sample_win_rate_bounds': [sum(x[j] for x in limits) / len(limits) for j in (0, 1)]}
    return {'purpose': 'Exploratory analysis specified after final outcomes were inspected; not preregistered confirmation',
            'paired_tests': results, 'censoring_sensitivity': bounds,
            'method': 'Two-sided sign flips of request-level paired differences; 99999 resamples, seed 11; Bonferroni over three contrasts',
            'assumptions': [
                'Under the null, method labels must be exchangeable within each request block (symmetric difference distribution).',
                'Request blocks are treated as independent; shared source sets, candidates and opponents can induce residual dependence.',
                'All repeated trials and orientations remain within their request; only trials decisive for both methods enter a contrast.',
                'Missing outcomes need not be random. Sensitivity bounds assign every missing result to a loss or win and are not confidence intervals.',
                'Failure to reject does not establish equivalence. P-values do not measure the probability that the null hypothesis is true.',
                'No ordinary t-test on overlapping cross-validation folds is performed.'],
            'references': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html',
                           'https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/final_statistical_tests.json')
    args = parser.parse_args()
    evaluation_path = ROOT / 'reports/final_evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    for relative, digest in evaluation['hashes'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'Frozen evaluation input changed: {relative}')
    rows = [json.loads(line) for line in (ROOT / 'data/processed/final_v1/labels.jsonl').read_text().splitlines()]
    report = analyze(rows, evaluation)
    report['hashes'] = {**evaluation['hashes'], **{relative: sha256(ROOT / relative) for relative in (
        'reports/final_evaluation.json', 'scripts/analyze_final_statistics.py', 'src/icon_pokemon/evaluation.py')}}
    report['versions'] = {name: version(name) for name in ('numpy', 'scipy')}
    save_json(args.output, report)
    print(json.dumps({'tests': report['paired_tests'], 'bounds': report['censoring_sensitivity']}, indent=2))


if __name__ == '__main__':
    main()
