"""Budgeted constructors with independently checked, validated success traces."""
from random import Random
from subprocess import TimeoutExpired
from time import perf_counter

from .matchups import coverage_score
from .optimization import solve_team


def quality(ids, request, matrix):
    """Maximize this tuple, in order; no fitted weights."""
    original = set(request.get('original', ()))
    return (len(original.intersection(ids)), *coverage_score(ids, matrix))


def admissible(ids, records, evidence, request, matrix):
    ids = list(ids)
    if len(ids) != request.get('size', 6) or len(set(ids)) != len(ids):
        return False
    by_id = {r['id']: r for r in records}
    if not set(ids) <= by_id.keys() or not set(request.get('fixed', ())) <= set(ids):
        return False
    if len({by_id[i]['base_species'] for i in ids}) != len(ids):
        return False
    capabilities = {c for i in ids for c in evidence[i]}
    if not set(request.get('required', ())) <= capabilities:
        return False
    offense, defense, _ = coverage_score(ids, matrix)
    return (offense >= request.get('minimum_offensive', 0)
            and defense >= request.get('minimum_defensive', 0))


def success_curve(runs, checkpoints, *, target=None):
    """All runs remain in the denominator, including errors and no-solution runs."""
    if not runs:
        raise ValueError('No runs')
    if any(t < 0 or any(t > run['budget_seconds'] for run in runs) for t in checkpoints):
        raise ValueError('Cannot extrapolate outside the common observed budget')
    return [{'seconds': t, 'successes': sum(any(
        event['seconds'] <= t and (target is None or tuple(event['quality']) >= tuple(target))
        for event in run['trace']) for run in runs), 'runs': len(runs)} for t in checkpoints]


def benchmark_run(method, records, evidence, request, matrix, validate, *, seed, budget):
    """validate(ids, remaining_seconds) must return Showdown's legality result.

    Random: repeated species-distinct samples, uniformly choosing a retention count
    for repairs. Greedy: repeated greedy constructions with random tie breaks.
    CP-SAT: observe incumbents; exclude any rejected complete teams and re-solve.
    """
    if method not in ('random', 'greedy', 'cp_sat') or budget <= 0:
        raise ValueError('Unknown method or invalid budget')
    started = perf_counter()
    deadline = started + budget
    rng = Random(seed)
    by_id = {r['id']: r for r in records}
    fixed = sorted(request.get('fixed', ()))
    original = set(request.get('original', ()))
    required = set(request.get('required', ()))
    size = request.get('size', 6)
    if not set(fixed).union(original) <= by_id.keys() or len(set(fixed)) != len(fixed):
        raise ValueError('Invalid fixed/original identifiers')
    if len(fixed) > size or len({by_id[i]['base_species'] for i in fixed}) != len(fixed):
        raise ValueError('Contradictory fixed members')
    trace, rejected, validations, diagnostics = [], [], {}, []
    attempts = 0
    status, certificate, error = 'BUDGET', None, None

    def observe(ids):
        nonlocal attempts
        attempts += 1
        key = tuple(sorted(ids))
        if not admissible(key, records, evidence, request, matrix):
            return False
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return False
        if key not in validations:
            validation = validate(list(key), remaining)
            validations[key] = validation
            if not validation['legal']:
                rejected.append({'ids': list(key), 'errors': validation.get('errors', [])})
        if not validations[key]['legal'] or perf_counter() > deadline:
            return False
        score = quality(key, request, matrix)
        if not trace or score > tuple(trace[-1]['quality']):
            trace.append({'seconds': perf_counter() - started, 'ids': list(key),
                          'quality': list(score),
                          'replacements': len(original - set(key)) if original else None})
        return True

    try:
        if method == 'cp_sat':
            excluded = []
            while perf_counter() < deadline:
                info = {}

                def callback(ids, objective, bound, elapsed):
                    observe(ids)
                    return perf_counter() < deadline

                result = solve_team(records, evidence, matchups=matrix, seed=seed,
                                    time_limit=max(1e-6, deadline - perf_counter()),
                                    excluded=excluded, on_solution=callback, diagnostics=info,
                                    **request)
                diagnostics.append({**info, 'status': result.status,
                                    'objective': result.objective, 'best_bound': result.best_bound})
                final = tuple(result.selected_ids)
                valid = bool(final and observe(final))
                if valid:
                    status = result.status
                    if result.status == 'OPTIMAL' and perf_counter() <= deadline:
                        certificate = {'seconds': perf_counter() - started,
                                       'quality': list(quality(final, request, matrix)),
                                       'objective': result.objective, 'bound': result.best_bound}
                    break
                if result.status == 'INFEASIBLE':
                    status = 'INFEASIBLE'
                    break
                if not final or final not in validations or validations[final]['legal']:
                    break
                excluded.append(list(final))
        else:
            groups = {}
            for r in sorted(records, key=lambda r: r['id']):
                groups.setdefault(r['base_species'], []).append(r['id'])
            while perf_counter() < deadline:
                chosen = list(fixed)
                if method == 'random' and original:
                    retainable = sorted(original - set(chosen))
                    chosen += rng.sample(retainable, rng.randrange(len(retainable) + 1))
                while len(chosen) < size and perf_counter() < deadline:
                    occupied = {by_id[i]['base_species'] for i in chosen}
                    free_species = sorted(set(groups) - occupied)
                    if not free_species:
                        break
                    if method == 'random':
                        chosen.append(rng.choice(groups[rng.choice(free_species)]))
                    else:
                        best, tied = None, []
                        for species in free_species:
                            for identifier in groups[species]:
                                if perf_counter() >= deadline:
                                    break
                                proposal = chosen + [identifier]
                                present = {c for i in proposal for c in evidence[i]}
                                score = (len(required & present), *quality(proposal, request, matrix))
                                if best is None or score > best:
                                    best, tied = score, [identifier]
                                elif score == best:
                                    tied.append(identifier)
                            if perf_counter() >= deadline:
                                break
                        if not tied or perf_counter() >= deadline:
                            break
                        chosen.append(rng.choice(tied))
                if len(chosen) == size:
                    observe(chosen)
                elif perf_counter() < deadline:
                    status = 'NO_CONSTRUCTION'
                    break
    except TimeoutExpired:
        status = 'VALIDATION_TIMEOUT'
    except Exception as exc:
        status, error = 'ERROR', f'{type(exc).__name__}: {exc}'
    elapsed = perf_counter() - started
    return {'method': method, 'seed': seed, 'budget_seconds': budget,
            'wall_seconds': elapsed, 'status': status, 'error': error,
            'attempts': attempts, 'unique_validations': len(validations),
            'trace': trace, 'first_success_seconds': trace[0]['seconds'] if trace else None,
            'best': trace[-1] if trace else None, 'certificate': certificate,
            'rejected': rejected, 'solver_diagnostics': diagnostics}
