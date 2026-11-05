"""Materialize OWL RL entailments, preserving the graph of asserted facts."""
from pathlib import Path
import hashlib
import json
from rdflib import Graph, Namespace, RDF, Literal, RDFS
from owlrl import DeductiveClosure, OWLRL_Semantics
from .move_semantics import EXTENDED_CAPABILITIES, import_move_semantics

P = Namespace("urn:icon:pokemon:")
ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = {
    "recovery": (P.SetWithRecoveryMove, P.SelfRecoveryMove),
    "hazard_setting": (P.SetWithHazardSettingMove, P.HazardSettingMove),
    "hazard_removal": (P.SetWithHazardRemovalMove, P.HazardRemovalMove),
    "priority_attack": (P.SetWithPriorityAttack, P.PriorityAttackMove),
}


TEAM_CAPABILITIES = dict(zip(CAPABILITIES, (P.TeamWithRecovery, P.TeamWithHazardSetting,
                                          P.TeamWithHazardRemoval, P.TeamWithPriorityAttack)))
ALL_CAPABILITIES = {**CAPABILITIES, **{
    key: (P['SetWith' + suffix], P[move_class])
    for key, (move_class, suffix) in EXTENDED_CAPABILITIES.items()}}
TEAM_CAPABILITIES.update({key: P['TeamWith' + suffix]
                         for key, (_, suffix) in EXTENDED_CAPABILITIES.items()})


def team_node(team):
    identifier = team.get('id') or hashlib.sha256(
        json.dumps(sorted(team['set_ids'])).encode()).hexdigest()[:24]
    return P['team/' + identifier]


def build_graph(catalog, mechanics=None, *, move_semantics=None):
    g = Graph().parse(ROOT / "ontology/pokemon.ttl")
    g.bind("p", P)
    annotations = json.loads((ROOT / "ontology/move_annotations.json").read_text())
    for record in catalog["sets"]:
        subject = P["set/" + record["id"]]
        g.add((subject, RDF.type, P.PokemonSet))
        g.add((subject, P.hasSpecies, P["species/" + record["species_id"]]))
        g.add((subject, P.hasAbility, P["ability/" + record["ability_id"]]))
        for type_name in record.get('types', []):
            type_node = P['type/' + type_name]
            g.add((type_node, RDF.type, P.Type))
            g.add((type_node, P.typeName, Literal(type_name)))
            g.add((subject, P.hasType, type_node))
        if record["set"].get("item"):
            g.add((subject, P.hasItem, P["item/" + record["item_id"]]))
        for source in record["provenance"]:
            g.add((subject, P.sourceFile, Literal(source)))
        for move in record["moves"]:
            node = P["move/" + move["id"]]
            g.add((subject, P.hasMove, node))
            g.add((node, RDF.type, P.Move))
            g.add((node, RDFS.label, Literal(move["name"])))
            if move.get('type'):
                g.add((node, P.hasType, P['type/' + move['type']]))
            g.add((node, P.moveCategory, Literal(move['category'])))
            if move["target"] == "self":
                g.add((node, P.targets, P.Self))
            if move["heal"] or move["id"] in annotations["direct_self_recovery_callbacks"]:
                g.add((node, P.hasEffect, P.DirectHealing))
                if not move["heal"]:
                    g.add((node, P.annotationSource, Literal(annotations["source"])))
            if move["side_condition"] in {"stealthrock", "spikes", "toxicspikes", "stickyweb"}:
                g.add((node, P.hasEffect, P.SetsHazards))
            if move["id"] in annotations["hazard_removal"]:
                g.add((node, P.hasEffect, P.RemovesHazards))
                g.add((node, P.annotationSource, Literal(annotations["source"])))
            if move["priority"] > 0 and move["category"] in {"Physical", "Special"}:
                g.add((node, P.hasEffect, P.PositivePriorityAttack))
    records = {r['id'] for r in catalog['sets']}
    teams = catalog.get('teams', [])
    for team in teams.values() if isinstance(teams, dict) else teams:
        if not team.get('legal', True):
            continue
        if not set(team['set_ids']) <= records:
            raise ValueError('Team refers to sets absent from the catalog')
        subject = team_node(team)
        g.add((subject, RDF.type, P.Team))
        for identifier in team['set_ids']:
            g.add((subject, P.hasMember, P['set/' + identifier]))
        if team.get('source'):
            g.add((subject, P.sourceFile, Literal(team['source'])))
    if mechanics:
        if mechanics['format'] != catalog.get('format') or mechanics['showdown_version'] != catalog.get('showdown_version'):
            raise ValueError('Mechanics and catalog must use the same format and Showdown version')
        for attack, defenses in mechanics['chart'].items():
            for defense, multiplier in defenses.items():
                relation = P[f'type-interaction/{attack}/{defense}']
                g.add((relation, RDF.type, P.TypeInteraction))
                g.add((relation, P.attackType, P['type/' + attack]))
                g.add((relation, P.defenseType, P['type/' + defense]))
                g.add((relation, P.multiplier, Literal(multiplier)))
    if move_semantics is not None:
        g.parse(ROOT / 'ontology/move_semantics.ttl')
        import_move_semantics(g, catalog, move_semantics)
    asserted = Graph()
    for triple in g:
        asserted.add(triple)
    DeductiveClosure(OWLRL_Semantics).expand(g)
    return asserted, g


def capability_evidence(graph, record):
    subject = P["set/" + record["id"]]
    evidence = {}
    for capability, (set_class, move_class) in ALL_CAPABILITIES.items():
        if (subject, RDF.type, set_class) in graph:
            evidence[capability] = [m["id"] for m in record["moves"]
                if (P["move/" + m["id"]], RDF.type, move_class) in graph]
    return evidence


def team_capability_evidence(graph, team):
    """Return member/move witnesses for entailed team capabilities."""
    subject = team_node(team)
    result = {}
    for capability, team_class in TEAM_CAPABILITIES.items():
        if (subject, RDF.type, team_class) not in graph:
            continue
        move_class = ALL_CAPABILITIES[capability][1]
        result[capability] = sorted(
            ({'set_id': str(member).split('/')[-1], 'move_id': str(move).split('/')[-1]}
             for member in graph.objects(subject, P.hasMember)
             for move in graph.objects(member, P.hasMove)
             if (move, RDF.type, move_class) in graph),
            key=lambda witness: (witness['set_id'], witness['move_id']))
    return result
