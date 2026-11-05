"""Enrich training sets only, demonstrate semantic requirements and preserve inputs."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.knowledge import (ALL_CAPABILITIES, P, build_graph,
                                    capability_evidence, team_capability_evidence, team_node)
from icon_pokemon.move_semantics import EXTENDED_CAPABILITIES, structured_capability_evidence
from icon_pokemon.optimization import solve_team
from rdflib import RDF


def main():
    started = perf_counter()
    path = ROOT / 'data/processed/dataset_v1/catalog.json'
    before = sha256(path)
    full = json.loads(path.read_text())
    manifest_path = ROOT / 'config/dataset_v1_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    teams = [t for t in manifest['teams'].values() if t['split'] == 'train']
    ids = {i for t in teams for i in t['set_ids']}
    records = [r for r in full['sets'] if r['id'] in ids]
    by_id = {r['id']: r for r in records}
    catalog = {**full, 'sets': records, 'teams': teams}
    semantics = showdown('export-move-semantics.cjs', {'format': full['format'],
        'move_ids': sorted({m['id'] for r in records for m in r['moves']})})
    output = ROOT / 'data/processed/kb_v2'
    save_json(output / 'move_semantics.json', semantics)
    inference_started = perf_counter()
    asserted, inferred = build_graph(catalog, move_semantics=semantics)
    inference_seconds = perf_counter() - inference_started
    evidence = {r['id']: capability_evidence(inferred, r) for r in records}
    required = ['physical_setup', 'screens']
    original = next(t for t in sorted(teams, key=lambda t: t['id'])
                    if not any('screens' in evidence[i] for i in t['set_ids']))
    repair = solve_team(records, evidence, original=original['set_ids'],
                        fixed=original['set_ids'][:2], required=required)
    if not repair.selected_ids:
        raise RuntimeError(f'KB repair demo failed: {repair.status}')
    validation = showdown('validate-team.cjs', {'format': full['format'],
        'sets': [by_id[i]['set'] for i in repair.selected_ids]})
    if not validation['legal']:
        raise RuntimeError(validation['errors'])
    witnesses = {i: {c: structured_capability_evidence(inferred, by_id[i], c, ALL_CAPABILITIES[c][1])
                    for c in required if c in evidence[i]} for i in repair.selected_ids}
    asserted.serialize(output / 'asserted.ttl', format='turtle')
    inferred.serialize(output / 'inferred.ttl', format='turtle')
    # No supervision is fitted and no held-out team or outcome is consumed here.
    assert sha256(path) == before
    report = {
        'scope': 'Training catalog only; semantic integration demo, not competitive evaluation',
        'format': full['format'], 'showdown_version': full['showdown_version'],
        'catalog_sha256': before, 'manifest_sha256': sha256(manifest_path),
        'teams': len(teams), 'sets': len(records), 'moves': len(semantics['moves']),
        'asserted_triples': len(asserted), 'closure_triples': len(inferred),
        'new_triples': len(inferred) - len(asserted), 'inference_seconds': inference_seconds,
        'set_capability_counts': dict(Counter(c for e in evidence.values() for c in e)),
        'team_capability_counts': dict(Counter(c for t in teams for c in team_capability_evidence(inferred, t))),
        'team_combinations': {name: sum((team_node(t), RDF.type, P[name]) in inferred for t in teams)
                              for name in ('TeamWithPhysicalSetupAndScreens', 'TeamWithSpecialSetupAndScreens')},
        'moves_with_structured_effects': sum(bool(m['effects']) for m in semantics['moves'].values()),
        'moves_with_uninterpreted_callbacks': {i: m['callbacks'] for i, m in semantics['moves'].items() if m['callbacks']},
        'callback_moves_without_structured_effects': [i for i, m in semantics['moves'].items() if m['callbacks'] and not m['effects']],
        'repair': {**asdict(repair), 'required': required, 'original_ids': original['set_ids'],
                   'fixed_ids': original['set_ids'][:2], 'showdown_legal': True,
                   'added': sorted(set(repair.selected_ids) - set(original['set_ids'])),
                   'removed': sorted(set(original['set_ids']) - set(repair.selected_ids)),
                   'witnesses': witnesses},
        'limits': ['Potential capabilities only; accuracy, immunity, costs and callback conditions can prevent activation.',
                   'Missing entailment is not a proof of absence. Items and abilities are not interpreted.',
                   'Original 236-feature supervised schema is unchanged. This is a versioned KB extension.',
                   'Closure size includes OWL/RDFS bookkeeping, not only useful domain deductions.'],
        'hashes': {p: sha256(ROOT / p) for p in ['ontology/pokemon.ttl', 'ontology/move_semantics.ttl',
            'ontology/move_annotations.json', 'tools/showdown/export-move-semantics.cjs',
            'src/icon_pokemon/knowledge.py', 'src/icon_pokemon/move_semantics.py',
            'src/icon_pokemon/optimization.py', 'tools/showdown/package-lock.json',
            'scripts/run_kb_enrichment.py']},
        'elapsed_seconds': perf_counter() - started,
    }
    save_json(ROOT / 'reports/kb_enrichment.json', report)
    print(json.dumps({k: report[k] for k in ['teams', 'sets', 'moves', 'asserted_triples',
        'closure_triples', 'inference_seconds', 'set_capability_counts', 'team_combinations']}, indent=2))
    print(json.dumps({'repair_status': repair.status, 'replacements': repair.replacements,
                      'legal': True}, indent=2))


if __name__ == '__main__':
    main()
