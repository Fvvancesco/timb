from itertools import combinations
import pytest

from icon_pokemon.solver_benchmark import benchmark_run, admissible, quality, success_curve


def problem():
    records = [{'id': x, 'base_species': x.upper()} for x in 'abcd']
    evidence = {'a': {}, 'b': {}, 'c': {'screens': ['reflect']}, 'd': {'screens': ['reflect']}}
    matrix = {x: {'threat': {'offensive': x in 'bd', 'defensive': x in 'ac'}} for x in 'abcd'}
    request = {'size': 2, 'fixed': ['a'], 'original': ['a', 'b'], 'required': ['screens']}
    return records, evidence, request, matrix


def test_cp_trace_matches_exhaustive_oracle_and_validates_before_success():
    records, evidence, request, matrix = problem()
    checked = []
    def validate(ids, remaining):
        checked.append(tuple(ids))
        return {'legal': True}
    run = benchmark_run('cp_sat', records, evidence, request, matrix, validate, seed=11, budget=2)
    feasible = [ids for ids in combinations('abcd', 2)
                if 'a' in ids and set(ids) & {'c', 'd'}]
    def independent_score(ids):
        return (len(set(ids) & {'a', 'b'}), int(bool(set(ids) & {'b', 'd'})),
                int(bool(set(ids) & {'a', 'c'})), min(2, len(set(ids) & {'b', 'd'})))
    assert tuple(run['best']['quality']) == max(map(independent_score, feasible))
    assert run['certificate'] is not None
    assert run['status'] == 'OPTIMAL'
    assert all(tuple(event['ids']) in checked for event in run['trace'])
    assert all(event['seconds'] <= run['budget_seconds'] for event in run['trace'])
    assert run['solver_diagnostics'][0]['variables'] > 0


def test_illegal_optimum_is_excluded_and_no_invalid_team_is_success():
    records, evidence, request, matrix = problem()
    def validate(ids, remaining):
        return {'legal': 'd' not in ids, 'errors': ['test exclusion'] if 'd' in ids else []}
    run = benchmark_run('cp_sat', records, evidence, request, matrix, validate, seed=11, budget=2)
    assert run['best']['ids'] == ['a', 'c']
    assert run['certificate'] is not None
    assert run['rejected']
    assert all('d' not in e['ids'] for e in run['trace'])


@pytest.mark.parametrize('method', ['random', 'greedy'])
def test_baselines_obey_the_same_hard_requirements(method):
    records, evidence, request, matrix = problem()
    run = benchmark_run(method, records, evidence, request, matrix,
                        lambda ids, remaining: {'legal': True}, seed=11, budget=0.05)
    assert run['trace']
    assert run['certificate'] is None
    for e in run['trace']:
        assert admissible(e['ids'], records, evidence, request, matrix)
        assert tuple(e['quality']) == quality(e['ids'], request, matrix)
    assert [e['quality'] for e in run['trace']] == sorted(e['quality'] for e in run['trace'])


def test_runtime_distribution_keeps_failures_and_distinguishes_quality():
    runs = [
        {'budget_seconds': 10, 'trace': [{'seconds': 1, 'quality': [2, 1]},
                                       {'seconds': 4, 'quality': [2, 3]}]},
        {'budget_seconds': 10, 'trace': [{'seconds': 7, 'quality': [1, 3]}]},
        {'budget_seconds': 10, 'trace': []},
    ]
    assert [v['successes'] for v in success_curve(runs, [0, 1, 5, 10])] == [0, 1, 1, 2]
    assert all(v['runs'] == 3 for v in success_curve(runs, [1, 10]))
    assert [v['successes'] for v in success_curve(runs, [1, 5, 10], target=[2, 3])] == [0, 1, 1]
    with pytest.raises(ValueError, match='extrapolate'):
        success_curve(runs, [11])
