"""Evidence-backed static type coverage, not a duel win-probability model."""
from math import prod

from .knowledge import P


def build_matchups(candidate_ids, opponent_ids, graph, mechanics):
    """Read types and move membership from the KB; multiply imported type facts."""
    candidate_ids, opponent_ids = list(candidate_ids), list(opponent_ids)
    if len(set(candidate_ids)) != len(candidate_ids) or len(set(opponent_ids)) != len(opponent_ids):
        raise ValueError('Duplicate set IDs in matchup axes')
    chart = {}
    for relation, multiplier in graph.subject_objects(P.multiplier):
        attack = graph.value(relation, P.attackType)
        defense = graph.value(relation, P.defenseType)
        chart[attack, defense] = float(multiplier)
    profiles = {}
    for identifier in sorted(set(candidate_ids + opponent_ids)):
        subject = P['set/' + identifier]
        types = set(graph.objects(subject, P.hasType))
        if not types:
            raise ValueError(f'Missing types for {identifier}')
        attacks, excluded = [], []
        for node in sorted(graph.objects(subject, P.hasMove), key=str):
            move_id = str(node).split('/')[-1]
            metadata = mechanics['moves'][move_id]
            if metadata['eligible']:
                move_types = list(graph.objects(node, P.hasType))
                if move_types != [P['type/' + metadata['type']]]:
                    raise ValueError(f'Inconsistent move type for {move_id}')
                attacks.append((move_id, move_types[0]))
            else:
                excluded.append({'move_id': move_id, 'reasons': metadata['excluded_reasons']})
        profiles[identifier] = (types, attacks, excluded)

    def evaluate(source, target):
        defenses = profiles[target][0]
        moves = [{'move_id': identifier,
                  'multiplier': prod(chart[attack, defense] for defense in defenses)}
                 for identifier, attack in profiles[source][1]]
        return moves, max((m['multiplier'] for m in moves), default=None)

    matrix = {}
    for candidate in candidate_ids:
        matrix[candidate] = {}
        for opponent in opponent_ids:
            outgoing, maximum = evaluate(candidate, opponent)
            incoming, incoming_maximum = evaluate(opponent, candidate)
            matrix[candidate][opponent] = {
                'offensive': maximum is not None and maximum > 1,
                'defensive': incoming_maximum is not None and incoming_maximum <= 1,
                'max_outgoing_multiplier': maximum,
                'max_incoming_multiplier': incoming_maximum,
                'offensive_witnesses': [m for m in outgoing if m['multiplier'] > 1],
                'incoming_evidence': incoming,
                'excluded_outgoing': profiles[candidate][2],
                'excluded_incoming': profiles[opponent][2],
            }
    return matrix


def coverage_score(selected_ids, matrix):
    """Lexicographic (offense, defense, capped offensive redundancy)."""
    threats = next(iter(matrix.values()), {})
    offense = [sum(matrix[i][t]['offensive'] for i in selected_ids) for t in threats]
    defense = [any(matrix[i][t]['defensive'] for i in selected_ids) for t in threats]
    return sum(n > 0 for n in offense), sum(defense), sum(min(n, 2) for n in offense)
