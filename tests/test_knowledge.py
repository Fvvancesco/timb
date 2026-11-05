from rdflib import RDF

from icon_pokemon.knowledge import P, build_graph, capability_evidence, team_capability_evidence, team_node


def record(identifier, move_id, *, target="self", heal=None, priority=0, category="Status", side_condition=None):
    return {"id": identifier, "species_id": identifier, "ability_id": "pressure",
            "item_id": "", "set": {"item": ""}, "provenance": ["synthetic"],
            "moves": [{"id": move_id, "name": move_id, "target": target,
                       "heal": heal, "priority": priority, "category": category,
                       "side_condition": side_condition}]}


def test_competency_questions_require_entailment():
    records = [record("recover", "recover", heal=[1, 2]),
               record("rest", "rest"), record("moonlight", "moonlight"),
               record("wish", "wish"), record("protect", "protect", priority=4),
               record("courtchange", "courtchange", target="all"),
               record("spin", "rapidspin", target="normal", category="Physical"),
               record("rock", "stealthrock", target="foeSide", side_condition="stealthrock"),
               record("quick", "quickattack", target="normal", priority=1, category="Physical")]
    asserted, inferred = build_graph({"sets": records})
    for identifier in ("recover", "rest", "moonlight"):
        claim = (P["set/" + identifier], RDF.type, P.SetWithRecoveryMove)
        assert claim not in asserted
        assert claim in inferred
    evidence = {r["id"]: capability_evidence(inferred, r) for r in records}
    assert evidence["recover"]["recovery"] == ["recover"]
    assert evidence["rest"]["recovery"] == ["rest"]
    assert evidence["moonlight"]["recovery"] == ["moonlight"]
    # Under the open-world assumption these are missing entailments, not proofs of negation.
    assert "recovery" not in evidence["wish"]
    assert "priority_attack" not in evidence["protect"]
    assert "hazard_removal" not in evidence["courtchange"]
    assert evidence["spin"]["hazard_removal"] == ["rapidspin"]
    assert evidence["rock"]["hazard_setting"] == ["stealthrock"]
    assert evidence["quick"]["priority_attack"] == ["quickattack"]


def test_team_inference_composes_member_move_effect_and_intersection():
    setter = record('setter', 'spikes', target='foeSide', side_condition='spikes')
    remover = record('remover', 'rapidspin', target='normal', category='Physical')
    team = {'id': 'support', 'set_ids': ['setter', 'remover']}
    asserted, inferred = build_graph({'sets': [setter, remover], 'teams': [team]})
    claim = (team_node(team), RDF.type, P.TeamWithHazardSupport)
    assert claim not in asserted
    assert claim in inferred
    assert (team_node(team), P.availableMove, P['move/rapidspin']) not in asserted
    assert (team_node(team), P.availableMove, P['move/rapidspin']) in inferred
    assert team_capability_evidence(inferred, team) == {
        'hazard_setting': [{'set_id': 'setter', 'move_id': 'spikes'}],
        'hazard_removal': [{'set_id': 'remover', 'move_id': 'rapidspin'}],
    }
    # Rebuild without the witness: absence of entailment, not an OWL negation.
    partial = {**team, 'set_ids': ['setter']}
    _, incomplete = build_graph({'sets': [setter, remover], 'teams': [partial]})
    assert (team_node(partial), RDF.type, P.TeamWithHazardSupport) not in incomplete
