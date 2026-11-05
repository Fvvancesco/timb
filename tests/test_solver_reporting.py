import pytest
from icon_pokemon.solver_reporting import summarize_runs, verify_run_set


def row(seed, time=None, score=1):
    event = {'seconds': time, 'quality': [score], 'ids': ['a']} if time is not None else None
    return {'instance_id': 'i', 'method': 'random', 'seed': seed, 'budget_seconds': 10,
            'status': 'BUDGET', 'trace': [event] if event else [], 'best': event, 'attempts': 1}


def test_percentiles_keep_censored_runs_and_unattained_quality():
    report = summarize_runs([row(1, 1), row(2), row(3), row(4, 4)], [2])
    valid = report['events']['valid']
    assert valid['quantiles_seconds'] == {'0.3': 4, '0.5': 4, '0.9': None}
    assert valid['curve'][-1]['fraction'] == .5
    assert report['events']['quality']['successes'] == 0
    assert report['events']['certificate'] is None


def test_reject_partial_duplicate_and_unexpected_results():
    runs = [row(1), row(2)]
    verify_run_set(runs, ['i'], ['random'], [1, 2])
    for invalid in (runs[:1], runs + [runs[0]], runs + [row(3)]):
        with pytest.raises(ValueError):
            verify_run_set(invalid, ['i'], ['random'], [1, 2])
