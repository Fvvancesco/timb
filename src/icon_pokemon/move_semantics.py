"""Import structured effects, retaining direction, timing and unsupported callbacks."""
from rdflib import Literal, RDF, Namespace

P = Namespace('urn:icon:pokemon:')
EXTENDED_CAPABILITIES = {
    'physical_setup': ('PhysicalSetupMove', 'PhysicalSetup'),
    'special_setup': ('SpecialSetupMove', 'SpecialSetup'),
    'speed_setup': ('SpeedSetupMove', 'SpeedSetup'),
    'defensive_setup': ('DefensiveSetupMove', 'DefensiveSetup'),
    'status_infliction': ('StatusInflictionMove', 'StatusInfliction'),
    'pivot': ('PivotMove', 'Pivot'),
    'forced_switch': ('ForcedSwitchMove', 'ForcedSwitch'),
    'screens': ('ScreenMove', 'Screens'),
}


def import_move_semantics(graph, catalog, semantics):
    if semantics.get('schema') != 'move-semantics-v1':
        raise ValueError('Unsupported move semantics schema')
    for field in ('format', 'showdown_version'):
        if semantics[field] != catalog.get(field):
            raise ValueError(f'Move semantics and catalog disagree on {field}')
    move_ids = {m['id'] for r in catalog['sets'] for m in r['moves']}
    if not move_ids <= semantics['moves'].keys():
        raise ValueError('Missing move semantics for catalog moves')
    for identifier in sorted(move_ids):
        move = P['move/' + identifier]
        data = semantics['moves'][identifier]
        graph.add((move, P.semanticsSource, Literal(
            f"Showdown {semantics['showdown_version']} / {semantics['format']}")))
        graph.add((move, P.activationScope, P.PotentialEffectOnly))
        for callback in data['callbacks']:
            graph.add((move, P.uninterpretedCallback, Literal(callback)))
        for index, effect in enumerate(data['effects']):
            node = P[f'move/{identifier}/effect/{index}']
            graph.add((move, P.hasStructuredEffect, node))
            graph.add((node, RDF.type, P.MoveEffect))
            graph.add((node, P.effectKind, P[effect['kind']]))
            graph.add((node, P.recipient, P[effect['recipient']]))
            graph.add((node, P.phase, P[effect['phase']]))
            graph.add((node, P.sourceField, Literal(effect['source_field'])))
            for field in ('stat', 'status', 'condition'):
                if field in effect:
                    graph.add((node, P[field], P[effect[field]]))
            for field in ('stages', 'chance', 'mode'):
                if effect.get(field) is not None:
                    graph.add((node, P[field], Literal(effect[field])))
            if 'stages' in effect:
                direction = 'Increase' if effect['stages'] > 0 else 'Decrease' if effect['stages'] < 0 else 'Unchanged'
                graph.add((node, P.direction, P[direction]))


def structured_capability_evidence(graph, record, capability, move_class):
    """Explain only effects that satisfy the inferred move capability."""
    effect_classes = {
        'physical_setup': P.PhysicalSetupEffect, 'special_setup': P.SpecialSetupEffect,
        'speed_setup': P.SpeedSetupEffect, 'defensive_setup': P.DefensiveSetupEffect,
        'status_infliction': P.StatusInflictionEffect, 'pivot': P.PivotEffect,
        'forced_switch': P.ForcedSwitchEffect, 'screens': P.ScreenEffect,
    }
    result = []
    for data in record['moves']:
        move = P['move/' + data['id']]
        if (move, RDF.type, move_class) not in graph:
            continue
        for effect in sorted(graph.objects(move, P.hasStructuredEffect), key=str):
            if (effect, RDF.type, effect_classes[capability]) not in graph:
                continue
            facts = {}
            for predicate in ('effectKind', 'recipient', 'phase', 'stat', 'direction',
                              'stages', 'chance', 'status', 'condition', 'mode', 'sourceField'):
                values = sorted(graph.objects(effect, P[predicate]), key=str)
                if values:
                    facts[predicate] = [v.toPython() if isinstance(v, Literal) else str(v).removeprefix(str(P)) for v in values]
            result.append({'move_id': data['id'], 'effect_id': str(effect), 'facts': facts,
                           'uninterpreted_callbacks': sorted(map(str, graph.objects(move, P.uninterpretedCallback))),
                           'scope': 'Potential capability; activation and success are not guaranteed.'})
    return result
