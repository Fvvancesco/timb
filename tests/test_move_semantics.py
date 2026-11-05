"""Competency questions against real fields in the pinned simulator."""
import pytest
from rdflib import RDF

from icon_pokemon.adapter import showdown
from icon_pokemon.knowledge import (P, ALL_CAPABILITIES, build_graph,
                                    capability_evidence, team_node)
from icon_pokemon.move_semantics import structured_capability_evidence
from icon_pokemon.optimization import solve_team
from icon_pokemon.features import STATS, TeamFeatures


@pytest.fixture(scope='module')
def example():
    ids = ['swordsdance', 'calmmind', 'shellsmash', 'flamecharge', 'uturn',
           'roar', 'willowisp', 'thunder', 'nuzzle', 'reflect', 'auroraveil',
           'bellydrum', 'growl', 'filletaway', 'protect', 'irondefense']
    semantics = showdown('export-move-semantics.cjs', {'format': 'gen9ou', 'move_ids': ids})
    records = [{
        'id': i, 'species_id': i, 'base_species': i, 'ability_id': 'pressure',
        'item_id': '', 'set': {}, 'provenance': ['competency-test'],
        'moves': [{'id': i, 'name': i, 'category': semantics['moves'][i]['category'],
                   'target': semantics['moves'][i]['target'], 'heal': None,
                   'priority': 0, 'side_condition': None}],
    } for i in ids]
    team = {'id': 'setup-screens', 'set_ids': ['swordsdance', 'reflect']}
    catalog = {'format': 'gen9ou', 'showdown_version': semantics['showdown_version'],
               'sets': records, 'teams': [team]}
    asserted, inferred = build_graph(catalog, move_semantics=semantics)
    return catalog, semantics, asserted, inferred, team


def test_effect_inference_preserves_recipient_phase_and_direction(example):
    catalog, _, asserted, graph, _ = example
    evidence = {r['id']: capability_evidence(graph, r) for r in catalog['sets']}
    claim = (P['set/swordsdance'], RDF.type, P.SetWithPhysicalSetup)
    assert claim not in asserted and claim in graph
    assert 'physical_setup' in evidence['swordsdance']
    assert {'special_setup', 'defensive_setup'} <= evidence['calmmind'].keys()
    assert {'physical_setup', 'special_setup', 'speed_setup'} <= evidence['shellsmash'].keys()
    assert 'defensive_setup' not in evidence['shellsmash']  # lowers both defenses
    assert 'defensive_setup' in evidence['irondefense']
    assert 'speed_setup' not in evidence['flamecharge']  # secondary, after attacking
    assert 'physical_setup' not in evidence['growl']  # lowers target attack
    assert 'physical_setup' not in evidence['bellydrum']  # callback-only, unknown
    assert (P['set/shellsmash'], RDF.type, P.SetWithMixedSetup) in graph


def test_probabilistic_effects_and_callbacks_are_visible_in_explanation(example):
    catalog, semantics, _, graph, _ = example
    records = {r['id']: r for r in catalog['sets']}
    cls = ALL_CAPABILITIES['status_infliction'][1]
    thunder = structured_capability_evidence(graph, records['thunder'], 'status_infliction', cls)
    assert len(thunder) == 1  # no secondary/secondaries duplication
    assert thunder[0]['facts']['chance'] == [30]
    assert thunder[0]['facts']['phase'] == ['secondary']
    assert thunder[0]['facts']['recipient'] == ['target']
    assert thunder[0]['uninterpreted_callbacks']
    nuzzle = structured_capability_evidence(graph, records['nuzzle'], 'status_infliction', cls)
    assert nuzzle[0]['facts']['chance'] == [100]
    assert 'not guaranteed' in nuzzle[0]['scope']
    assert 'onHit' in semantics['moves']['bellydrum']['callbacks']
    veil = structured_capability_evidence(graph, records['auroraveil'], 'screens', P.ScreenMove)
    assert 'onTry' in veil[0]['uninterpreted_callbacks']
    costly = structured_capability_evidence(graph, records['filletaway'], 'physical_setup', P.PhysicalSetupMove)
    assert costly[0]['uninterpreted_callbacks']  # HP prerequisites/cost not reconstructed


def test_relational_composition_and_removal_of_premise(example):
    catalog, semantics, asserted, graph, team = example
    claim = (team_node(team), RDF.type, P.TeamWithPhysicalSetupAndScreens)
    assert claim not in asserted and claim in graph
    partial = {**catalog, 'teams': [{**team, 'set_ids': ['swordsdance']}]}
    _, graph = build_graph(partial, move_semantics=semantics)
    assert claim not in graph


def test_new_inferences_drive_minimal_repair(example):
    catalog, _, _, graph, _ = example
    records = [r for r in catalog['sets'] if r['id'] in ('swordsdance', 'reflect', 'protect')]
    evidence = {r['id']: capability_evidence(graph, r) for r in records}
    result = solve_team(records, evidence, size=2, original=['swordsdance', 'protect'],
                        fixed=['swordsdance'], required=['physical_setup', 'screens'])
    assert result.status == 'OPTIMAL'
    assert result.replacements == 1
    assert result.selected_ids == ['reflect', 'swordsdance']
    # Without the inferred premises the same request cannot be certified.
    result = solve_team(records, {r['id']: {} for r in records}, size=2,
                        required=['physical_setup', 'screens'])
    assert result.status == 'INFEASIBLE'


def test_sidecar_rejects_wrong_version_or_incomplete_coverage(example):
    catalog, semantics, *_ = example
    with pytest.raises(ValueError, match='showdown_version'):
        build_graph(catalog, move_semantics={**semantics, 'showdown_version': 'wrong'})
    with pytest.raises(ValueError, match='Missing move semantics'):
        build_graph(catalog, move_semantics={**semantics, 'moves': {}})


def test_extended_features_preserve_original_values_and_pair_symmetry(example):
    catalog, _, _, original_graph, _ = example
    graph = original_graph.__class__()
    for triple in original_graph:
        graph.add(triple)
    records = {r['id']: {**r, 'types': ['Normal'], 'base_stats': dict.fromkeys(STATS, 80)}
               for r in catalog['sets']}
    for i in records:
        graph.add((P['set/' + i], P.hasType, P['type/Normal']))
    mechanics = {'types': ['Normal'], 'chart': {'Normal': {'Normal': 1}},
                 'natures': {'Serious': {'plus': None, 'minus': None}},
                 'moves': {i: {'eligible': False, 'excluded_reasons': ['test_scope']}
                           for i in records}}
    evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
    old = TeamFeatures(records, graph, mechanics, evidence)
    new = TeamFeatures(records, graph, mechanics, evidence, capabilities=ALL_CAPABILITIES)
    a, b = ['swordsdance', 'reflect'], ['calmmind', 'uturn']
    old_values = dict(zip(old.names, old.pair(a, b), strict=True))
    extended = new.pair(a, b)
    new_values = dict(zip(new.names, extended, strict=True))
    assert all(new_values[name] == value for name, value in old_values.items())
    assert len(new.names) == len(old.names) + 16
    assert new_values['a_physical_setup'] == new_values['a_screens'] == 1
    assert new_values['b_pivot'] == new_values['b_special_setup'] == 1
    assert new.pair(a[::-1], b[::-1]) == extended
    halfway = len(extended) // 2
    assert new.pair(b, a) == extended[halfway:] + extended[:halfway]
    with pytest.raises(ValueError, match='distinct known'):
        TeamFeatures(records, graph, mechanics, evidence, capabilities=['made_up'])
