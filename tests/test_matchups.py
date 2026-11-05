import pytest

from icon_pokemon.adapter import showdown
from icon_pokemon.knowledge import build_graph
from icon_pokemon.matchups import build_matchups


@pytest.fixture(scope='module')
def static_case():
    moves = ['thunderbolt', 'earthquake', 'icebeam', 'seismictoss', 'terablast',
             'freezedry', 'flyingpress', 'recover', 'tackle']
    mechanics = showdown('export-mechanics.cjs', {'format': 'gen9ou', 'move_ids': moves})
    def record(identifier, types, move_ids):
        return {'id': identifier, 'species_id': identifier, 'ability_id': 'pressure',
                'item_id': '', 'set': {}, 'provenance': ['synthetic'], 'types': types,
                'moves': [{'id': m, 'name': m, 'type': mechanics['moves'][m]['type'],
                           'category': mechanics['moves'][m]['category'], 'target': 'normal',
                           'heal': None, 'side_condition': None, 'priority': 0} for m in move_ids]}
    records = [record('electric', ['Electric'], ['thunderbolt']),
               record('water_flying', ['Water', 'Flying'], ['tackle']),
               record('ground_flying', ['Ground', 'Flying'], ['earthquake']),
               record('ice', ['Ice'], ['icebeam']),
               record('ghost', ['Ghost'], ['recover']),
               record('special', ['Normal'], ['seismictoss', 'terablast', 'freezedry', 'flyingpress'])]
    _, graph = build_graph({'sets': records, 'format': 'gen9ou',
                            'showdown_version': mechanics['showdown_version']}, mechanics)
    ids = [r['id'] for r in records]
    return mechanics, build_matchups(ids, ids, graph, mechanics)


def test_dual_types_and_immunity_are_multiplied_not_added(static_case):
    _, matrix = static_case
    assert matrix['electric']['water_flying']['max_outgoing_multiplier'] == 4
    assert matrix['electric']['ground_flying']['max_outgoing_multiplier'] == 0
    assert matrix['ice']['ground_flying']['max_outgoing_multiplier'] == 4
    assert matrix['water_flying']['ghost']['max_outgoing_multiplier'] == 0
    assert matrix['electric']['water_flying']['offensive_witnesses'] == [
        {'move_id': 'thunderbolt', 'multiplier': 4}]


def test_unknown_or_excluded_attacks_do_not_certify_defensive_coverage(static_case):
    mechanics, matrix = static_case
    for move in ('seismictoss', 'terablast', 'freezedry', 'flyingpress', 'recover'):
        assert not mechanics['moves'][move]['eligible']
    for opponent in ('ghost', 'special'):
        assert matrix['electric'][opponent]['max_incoming_multiplier'] is None
        assert matrix['electric'][opponent]['defensive'] is False


def test_mechanics_version_mismatch_is_rejected(static_case):
    mechanics, _ = static_case
    with pytest.raises(ValueError, match='same format'):
        build_graph({'sets': [], 'format': 'gen8ou', 'showdown_version': 'wrong'}, mechanics)
