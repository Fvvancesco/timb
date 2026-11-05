"""Helpers for preserving incomplete pilot battles without draw labels."""
import re

def team_text(text):
    return text.strip()

def foul_result(log, names, *, timed_out=False, process_failed=False, forced_stop=None, cap=200):
    turns = [int(x) for x in re.findall(r'Turn: (\d+)', log)]
    turn = max(turns, default=0)
    if timed_out or process_failed:
        return {'status':'timeout' if timed_out else 'error', 'outcome':None, 'turn':turn}
    if forced_stop:
        return {'status':'truncated' if forced_stop == 'turn_limit' else forced_stop, 'outcome':None, 'turn':turn}
    winner = re.search(r'Winner: (\S+)', log)
    if winner and winner.group(1).lower() in [n.lower() for n in names] and turn <= cap:
        return {'status':'completed', 'outcome':f'p{[n.lower() for n in names].index(winner.group(1).lower())+1}', 'turn':turn}
    if re.search(r'^\|tie(?:\||$)', log, re.MULTILINE):
        return {'status':'completed', 'outcome':'draw', 'turn':turn}
    return {'status':'running', 'outcome':None, 'turn':turn}

def public_result(log):
    players = dict(re.findall(r'^\|player\|p([12])\|([^|\n]+)\|', log, re.MULTILINE))
    winner = re.search(r'^\|win\|([^\n]+)', log, re.MULTILINE)
    tie = re.search(r'^\|tie(?:\||$)', log, re.MULTILINE)
    return {'players': {f'p{k}': v for k, v in players.items()},
            'winner': winner.group(1) if winner else None, 'tie': bool(tie)}

def showdown_seed(seed):
    if not isinstance(seed, list) or len(seed) != 4 or any(type(x) is not int or not 0 <= x <= 65535 for x in seed):
        raise ValueError('Expected four unsigned 16-bit seed words')
    return 'gen5,' + ''.join(f'{x:04x}' for x in seed)

def team_differences(expected, actual):
    """Check battle-relevant fields without conflating blank/default values."""
    def norm(value):
        return ''.join(c for c in str(value or '').lower() if c.isalnum())
    diffs=[]
    if len(expected)!=len(actual): return [f'team length {len(expected)} != {len(actual)}']
    for index,(exp,got) in enumerate(zip(expected,actual)):
        for field in ('species','item','ability','nature','gender','teraType','hpType'):
            if norm(exp.get(field)) != norm(got.get(field)):
                diffs.append(f'{index}.{field}: {exp.get(field)!r} != {got.get(field)!r}')
        if norm(exp.get('name')) != norm(exp.get('species')) and norm(exp.get('name')) != norm(got.get('name')):
            diffs.append(f'{index}.name nickname mismatch')
        if [norm(v) for v in exp.get('moves',[])] != [norm(v) for v in got.get('moves',[])]:
            diffs.append(f'{index}.moves mismatch')
        for stats,default in (('evs',0),('ivs',31)):
            for stat in ('hp','atk','def','spa','spd','spe'):
                if exp.get(stats,{}).get(stat,default)!=got.get(stats,{}).get(stat,default):
                    diffs.append(f'{index}.{stats}.{stat} mismatch')
        for field,default in (('level',100),('happiness',255)):
            if exp.get(field,default)!=got.get(field,default): diffs.append(f'{index}.{field} mismatch')
    return diffs

def classify_engine_result(data, error):
    if error or not data:
        return 'error', None
    forced = data.get('forced_stop')
    if forced:
        return ('timeout' if forced == 'wall_timeout' else 'truncated' if forced == 'turn_limit' else 'error'), None
    return 'completed', data.get('winner') or 'draw'


def known_pilot_censor(result, client_logs):
    """Recognize the archived Foul Play zero-visit MCTS crash only."""
    if (result.get('status'), result.get('outcome'), result.get('error')) != (
        'error', None, "RuntimeError('Client exited before engine record')"
    ):
        return None
    markers = ('in select_move_from_mcts_results',
               'this_policy.total_score / this_policy.visits',
               'ZeroDivisionError: float division by zero')
    if any(all(marker in log for marker in markers) for log in client_logs):
        return 'foul_play_zero_visit_mcts'
    return None
