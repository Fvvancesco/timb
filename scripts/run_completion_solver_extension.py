"""Compare one versus five predicted species scenarios in constrained completion."""
import json
from math import prod
from pathlib import Path
from random import Random
from subprocess import TimeoutExpired
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.evaluation import grouped_interval
from icon_pokemon.optimization import solve_team
from icon_pokemon.validation import TeamValidator

OUTPUT = ROOT / 'reports/team_completion_v1'
CATALOG = ROOT / 'data/processed/dataset_v1/catalog.json'
SEED, REQUESTS, WEIGHT, BUDGET = 719, 100, 20, 2.


def scenario_weights(predictions):
    probabilities = [p for _, p in predictions]
    available = WEIGHT - len(predictions)
    quotas = [p / sum(probabilities) * available for p in probabilities]
    weights = [int(q) + 1 for q in quotas]
    for i in sorted(range(len(weights)), key=lambda i: -(quotas[i] % 1))[:WEIGHT-sum(weights)]:
        weights[i] += 1
    return [(name, count) for (name, _), count in zip(predictions, weights, strict=True)]


def solve(records, fixed, visible, predictions, profiles, validator, format_id):
    # Replicating threat columns represents integer weights in the existing solver.
    names = [name for name in visible for _ in range(WEIGHT)]
    names += [name for name, count in scenario_weights(predictions) for _ in range(count)]
    matrix = {r['id']: {str(i): {'offensive': name in profiles[r['id']], 'defensive': False}
                        for i, name in enumerate(names)} for r in records}
    evidence = {r['id']: () for r in records}
    by_id = {r['id']: r['set'] for r in records}
    deadline, rejected = perf_counter() + BUDGET, []
    try:
        while (remaining := deadline - perf_counter()) > 0:
            result = solve_team(records, evidence, fixed=fixed, matchups=matrix,
                                time_limit=remaining, excluded=rejected)
            if not result.selected_ids:
                return {'status': result.status, 'legal': False}
            validation = validator.validate(format_id, [by_id[i] for i in result.selected_ids],
                                            deadline - perf_counter())
            if validation['legal']:
                return {'status': result.status, 'legal': True, 'team': result.selected_ids,
                        'rejected': len(rejected)}
            rejected.append(result.selected_ids)
    except TimeoutExpired:
        pass
    return {'status': 'TIMEOUT', 'legal': False}


def main():
    prediction_file = OUTPUT / 'test_predictions.jsonl'
    selection_file = OUTPUT / 'selection.json'
    selection = json.loads(selection_file.read_text())
    method = min(('naive_bayes', 'pokemon2vec'), key=lambda m: (
        -selection[{'naive_bayes': 'bayes_selected', 'pokemon2vec': 'pokemon2vec_selected'}[m]]['top5'],
        selection[{'naive_bayes': 'bayes_selected', 'pokemon2vec': 'pokemon2vec_selected'}[m]]['log_loss_covered']))
    cases = {}
    for row in map(json.loads, prediction_file.read_text().splitlines()):
        cases.setdefault(row['team_id'], []).append(row)
    catalog = json.loads(CATALOG.read_text())
    own_teams = sorted((t for t in catalog['teams'].values() if t['split'] == 'train'), key=lambda t: t['id'])
    allowed = {i for t in own_teams for i in t['set_ids']}
    records = sorted((r for r in catalog['sets'] if r['id'] in allowed), key=lambda r: r['id'])
    rng = Random(SEED)
    requests = [(rng.choice(cases[i]), rng.sample(rng.choice(own_teams)['set_ids'], 5))
                for i in rng.sample(sorted(cases), REQUESTS)]
    names = sorted({name for row, _ in requests for name in
                    row['visible'] + [row['target']] + [n for n, _ in row['methods'][method]['top5']]})
    species = showdown('export-preview-species.cjs', {'names': names})
    if set(species) != set(names):
        raise ValueError('A base species could not be mapped to the pinned Dex')
    mechanics = showdown('export-mechanics.cjs', {'format': catalog['format'],
        'move_ids': sorted({m['id'] for r in records for m in r['moves']})})
    chart, moves = mechanics['chart'], mechanics['moves']
    profiles = {}
    for r in records:
        attack_types = {moves[m['id']]['type'] for m in r['moves'] if moves[m['id']]['eligible']}
        profiles[r['id']] = {name for name in names if any(
            prod(chart[a][d] for d in species[name]['types']) > 1 for a in attack_types)}
    runs = []
    with TeamValidator() as validator:
        for row, fixed in requests:
            results = {}
            for k in (1, 5):
                result = solve(records, fixed, row['visible'], row['methods'][method]['top5'][:k],
                               profiles, validator, catalog['format'])
                if result['legal']:
                    hits = [sum(name in profiles[i] for i in result['team'])
                            for name in [*row['visible'], row['target']]]
                    result.update(hidden_covered=int(hits[-1] > 0), hidden_responses=hits[-1],
                                  visible_covered=sum(n > 0 for n in hits[:5]))
                results[f'top{k}'] = result
            runs.append({'opponent_id': row['team_id'], 'visible': row['visible'],
                         'hidden': row['target'], 'fixed': fixed, **results})
    paired = [r for r in runs if r['top1']['legal'] and r['top5']['legal']]
    measures = ('hidden_covered', 'hidden_responses', 'visible_covered')
    report = {'method_selected_on_validation': method, 'requests': len(runs), 'paired_legal': len(paired),
              'scope': 'Exploratory follow-up using canonical base-species typing and static super-effective moves. No battle outcomes. Top-5 probabilities truncated and quantized to 20 integer weight units, at least one per scenario.',
              'seed': SEED, 'budget_seconds': BUDGET, 'measures': measures,
              'hashes': {str(p.relative_to(ROOT)): sha256(p) for p in (CATALOG, prediction_file, selection_file, Path(__file__))},
              'methods': {m: {'legal': sum(r[m]['legal'] for r in runs),
                              'optimal': sum(r[m]['legal'] and r[m]['status'] == 'OPTIMAL' for r in runs),
                              'means_paired': {k: sum(r[m][k] for r in paired)/len(paired) for k in measures} if paired else {}}
                          for m in ('top1','top5')},
              'paired_top5_minus_top1': grouped_interval([[r['top5'][k]-r['top1'][k] for k in measures] for r in paired],
                                                          [r['opponent_id'] for r in paired]),
              'runs': runs}
    save_json(OUTPUT / 'solver_scenarios.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
