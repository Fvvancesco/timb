"""Postprocessing only: censored empirical run-time distributions."""
from collections import Counter
from math import ceil


def event_time(run, target=None, *, certificate=False):
    if certificate:
        proof = run.get('certificate')
        return proof['seconds'] if proof and proof['seconds'] <= run['budget_seconds'] else None
    return next((event['seconds'] for event in run['trace']
                 if event['seconds'] <= run['budget_seconds']
                 and (target is None or tuple(event['quality']) >= tuple(target))), None)


def summarize_runs(runs, target):
    if not runs or len({r['budget_seconds'] for r in runs}) != 1:
        raise ValueError('Nonempty runs with a common cutoff required')
    n = len(runs)
    cutoff = runs[0]['budget_seconds']
    methods = {r['method'] for r in runs}
    if len(methods) != 1:
        raise ValueError('Summarize each method separately')
    result = {'runs': n, 'budget_seconds': cutoff,
              'statuses': dict(Counter(r['status'] for r in runs)),
              'events': {}}
    for name, kwargs in [('valid', {}), ('quality', {'target': target}), ('certificate', {'certificate': True})]:
        if name == 'certificate' and methods != {'cp_sat'}:
            result['events'][name] = None
            continue
        times = [event_time(run, **kwargs) for run in runs]
        observed = sorted(t for t in times if t is not None)
        # Denominator is ALL runs; a percentile may be unattained at the cutoff.
        quantiles = {}
        for probability in (.3, .5, .9):
            rank = ceil(probability * n)
            quantiles[str(probability)] = observed[rank - 1] if rank <= len(observed) else None
        result['events'][name] = {
            'successes': len(observed), 'failures_by_cutoff': n - len(observed),
            'quantiles_seconds': quantiles,
            'curve': [{'seconds': time, 'successes': sum(t <= time for t in observed),
                       'runs': n, 'fraction': sum(t <= time for t in observed) / n}
                      for time in sorted({0., cutoff, *observed})]}
    result['distinct_incumbent_sequences'] = len({tuple(
        (tuple(e['ids']), tuple(e['quality'])) for e in r['trace']) for r in runs})
    result['distinct_final_teams'] = len({tuple(r['best']['ids']) for r in runs if r.get('best')})
    result['attempts'] = {'min': min(r['attempts'] for r in runs),
                          'max': max(r['attempts'] for r in runs)}
    return result


def verify_run_set(runs, instances, methods, seeds):
    expected = {(i, m, s) for i in instances for m in methods for s in seeds}
    found = [(r['instance_id'], r['method'], r['seed']) for r in runs]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError('Incomplete, duplicated or unexpected solver runs')
