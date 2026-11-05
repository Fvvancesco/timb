import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('pilot_result_helpers', Path(__file__).resolve().parents[1] / 'tools/pilots/result_helpers.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def test_team_normalization_preserves_internal_separation():
    assert h.team_text('One\n\nTwo\n\n') == 'One\n\nTwo'

def test_win_and_explicit_draw():
    assert h.foul_result('Turn: 5\nWinner: BotB', ['BotA','BotB'])['outcome'] == 'p2'
    assert h.foul_result('Turn: 5\n|tie|', ['BotA','BotB'])['outcome'] == 'draw'

def test_incomplete_is_never_a_draw():
    for log, kwargs, status in [('Turn: 201', {'forced_stop':'turn_limit'}, 'truncated'), ('Turn: 7', {'timed_out':True}, 'timeout'), ('', {'process_failed':True}, 'error')]:
        result = h.foul_result(log, ['BotA','BotB'], **kwargs)
        assert result['status'] == status and result['outcome'] is None

def test_public_result_records_actual_showdown_positions():
    r = h.public_result('|player|p1|SecondBot|1|\n|player|p2|FirstBot|2|\n|win|FirstBot\n|player|p1|\n')
    assert r == {'players': {'p1':'SecondBot', 'p2':'FirstBot'}, 'winner':'FirstBot', 'tie':False}

def test_tier_is_not_tie_and_censoring_overrides_tie():
    assert not h.public_result('|tier|[Gen 9] OU\n')['tie']
    assert h.public_result('|tie\n')['tie']
    assert h.foul_result('|tie\n', ['A','B'])['outcome'] == 'draw'
    assert h.foul_result('|tie\n', ['A','B'], timed_out=True)['outcome'] is None
    assert h.foul_result('Turn: 201\n|tie\n', ['A','B'], forced_stop='turn_limit')['outcome'] is None


def test_natural_win_on_turn_200_is_not_censored():
    r = h.foul_result('Turn: 200\nWinner: A', ['A', 'B'])
    assert r['status'] == 'completed' and r['outcome'] == 'p1'
    assert h.foul_result('Turn: 200', ['A', 'B'])['status'] == 'running'

def test_seed_encoding_and_team_check():
    assert h.showdown_seed([62695,60486,14882,43623]) == 'gen5,f4e7ec463a22aa67'
    import pytest
    with pytest.raises(ValueError): h.showdown_seed([1,2,3,65536])
    mon={'name':'A', 'species':'A', 'moves':['Foo Bar'], 'evs':{}, 'ivs':{}}
    assert h.team_differences([mon],[dict(mon)]) == []
    changed=dict(mon, moves=['Other'])
    assert h.team_differences([mon],[changed])

def test_species_default_name_normalization_preserves_distinct_nicknames():
    base={'name':'Weezing-Galar','species':'Weezing-Galar','moves':[], 'evs':{},'ivs':{}}
    normalized=dict(base,name='Weezing')
    assert h.team_differences([base],[normalized]) == []
    nickname=dict(base,name='My Pal')
    assert h.team_differences([nickname],[normalized])

def test_engine_error_takes_precedence_over_forced_stop():
    assert h.classify_engine_result({'forced_stop':'turn_limit','winner':''}, 'seed mismatch') == ('error',None)
    assert h.classify_engine_result({'forced_stop':'turn_limit','winner':''}, None) == ('truncated',None)
    assert h.classify_engine_result({'forced_stop':None,'winner':'A'}, None) == ('completed','A')


def test_known_pilot_censor_requires_exact_crash_and_no_outcome():
    result = {'status': 'error', 'outcome': None,
              'error': "RuntimeError('Client exited before engine record')"}
    log = ('in select_move_from_mcts_results\n'
           'round(this_policy.total_score / this_policy.visits, 3),\n'
           'ZeroDivisionError: float division by zero')
    assert h.known_pilot_censor(result, [log]) == 'foul_play_zero_visit_mcts'
    assert h.known_pilot_censor(result, ['ZeroDivisionError: float division by zero']) is None
    assert h.known_pilot_censor({**result, 'error': 'Engine seed mismatch'}, [log]) is None
    assert h.known_pilot_censor({**result, 'outcome': 'p1'}, [log]) is None
