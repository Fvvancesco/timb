"""Generate one legal candidate pool and compare alternative selectors."""
from dataclasses import asdict
from random import Random
from subprocess import TimeoutExpired
from time import perf_counter

from .adapter import showdown
from .optimization import solve_team
from .supervised import probabilities


def candidate_pool(records, evidence, *, format, matchups, limit=20, time_limit=10, **request):
    if limit < 1 or time_limit <= 0:
        raise ValueError('Pool limit and time budget must be positive')
    start = perf_counter()
    by_id = {record['id']: record['set'] for record in records}
    teams, excluded, rejections = [], [], []
    replacement_cap = None
    stop = 'limit'
    while len(teams) < limit:
        remaining = time_limit - (perf_counter() - start)
        if remaining <= 0:
            stop = 'time_limit'
            break
        result = solve_team(records, evidence, matchups=matchups,
                            excluded=excluded, maximum_replacements=replacement_cap,
                            time_limit=remaining, **request)
        if not result.selected_ids:
            stop = result.status
            break
        excluded.append(result.selected_ids)
        remaining = time_limit - (perf_counter() - start)
        if remaining <= 0:
            stop = 'time_limit'
            break
        try:
            validation = showdown('validate-team.cjs', {'format': format, 'sets': [
                by_id[i] for i in result.selected_ids]}, timeout=remaining)
        except TimeoutExpired:
            stop = 'validation_timeout'
            break
        if not validation['legal']:
            rejections.append({'ids': result.selected_ids, 'errors': validation['errors']})
            continue
        teams.append({**asdict(result), 'packed': validation['packed']})
        if request.get('original') and replacement_cap is None:
            if result.status != 'OPTIMAL':
                stop = 'minimal_repair_unproven'
                break
            replacement_cap = result.replacements
    return {'teams': teams, 'rejections': rejections, 'stop': stop,
            'minimum_replacements_proven': replacement_cap,
            'wall_seconds': perf_counter() - start}


def select_candidates(pool, opponent, features, model, seed=11):
    if not pool:
        raise ValueError('Cannot select from an empty pool')
    ranked = sorted(pool, key=lambda team: tuple(team['selected_ids']))
    scores = probabilities(model, [features.pair(team['selected_ids'], opponent) for team in ranked])
    indices = {
        'random': Random(seed).randrange(len(ranked)),
        'symbolic': max(range(len(ranked)), key=lambda i: ranked[i]['objective']),
        'supervised': max(range(len(ranked)), key=lambda i: scores[i]),
    }
    return {
        'choices': {method: {'ids': ranked[i]['selected_ids'], 'model_probability': float(scores[i])}
                    for method, i in indices.items()},
        'scores': [{'ids': team['selected_ids'], 'model_probability': float(score)}
                   for team, score in zip(ranked, scores, strict=True)],
    }
