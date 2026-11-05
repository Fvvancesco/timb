"""Audit all panel battles and estimate seed and pilot sensitivity by team pair."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from export_simulated_labels import export_batch, sha256
from run_coverage_confirmation import ROOT, dump

FOUL = ROOT / 'data/processed/matchup_panel_v1'
SIMPLE = ROOT / 'data/processed/matchup_panel_simple_v1'


def load(path):
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    digest = sha256(manifest_path)
    assignments = [json.loads(p.read_text()) for p in sorted((path/'assignments').glob('*.json'))]
    ids = [i for a in assignments for i in a['requests']]
    if (len(ids) != len(set(ids)) or set(ids) != {r['id'] for r in manifest['requests']}
            or any(a['manifest_sha256'] != digest for a in assignments)):
        raise ValueError('Assignments do not cover the manifest exactly')
    jobs = {j['job_id']: j for j in manifest['jobs']}
    rows = {}
    for a in assignments:
        subset = {i: j for i, j in jobs.items() if j['request_id'] in a['requests']}
        for r in export_batch(path/a['shard']/'batch', subset, digest, manifest['source_sha256']['catalog.json']):
            if r['job_id'] in rows:
                raise ValueError('Duplicate battle')
            rows[r['job_id']] = r
    if set(rows) != set(jobs):
        raise ValueError(f'{len(set(jobs) - set(rows))} battles still missing in {path}')
    return manifest, digest, rows


def per_root(rows):
    sides = defaultdict(dict)
    for row in rows.values():
        if row['outcome'] is None:
            continue
        o = row['orientation']
        score = .5 if row['outcome'] == 'draw' else float(row['outcome'] == ('team_a' if o == 0 else 'team_b'))
        sides[row['request_id'], row['root']][o] = score
    result = defaultdict(dict)
    for (pair, seed), group in sides.items():
        if set(group) == {0, 1}:
            result[pair][seed] = sum(group.values())/2
    return result


def paired_summary(pairs):
    x = np.asarray(pairs, dtype=float)
    if len(x) < 2:
        raise ValueError('Too few usable pairs')
    d = x[:, 0]-x[:, 1]
    rng = np.random.default_rng(11)
    sampled = d[rng.integers(0, len(d), size=(10000,len(d)))].mean(axis=1)
    return {'pairs': len(x), 'mean_first': float(x[:,0].mean()), 'mean_second': float(x[:,1].mean()),
            'mean_signed_difference': float(d.mean()), 'mean_absolute_difference': float(np.abs(d).mean()),
            'interval_95_mean_signed_difference': np.quantile(sampled,[.025,.975]).tolist(),
            'pearson': float(np.corrcoef(x.T)[0,1]) if np.all(x.std(axis=0)>0) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foul-only', action='store_true')
    args = parser.parse_args()
    m, digest, foul = load(FOUL)
    roots = per_root(foul)
    halves = {}
    for row in m['requests']:
        pair = row['id']
        counts = [sum(seed in roots[pair] for seed in g) for g in m['root_groups']]
        means = [sum(roots[pair][seed] for seed in g if seed in roots[pair])/n if n else None
                 for g,n in zip(m['root_groups'], counts, strict=True)]
        halves[pair] = {'means': means, 'available_roots': counts}
    summary = {'manifest_sha256':digest, 'battles':len(foul),
               'statuses':dict(Counter(r['status'] for r in foul.values())),
               'seed_half_comparison': paired_summary([r['means'] for r in halves.values() if None not in r['means']]),
               'per_pair': halves,
               'scope': 'Rates for 30 pairs of development teams, both sides per seed. Conditional on simulation and chosen policies. Small per-pair replication makes split-half correlation descriptive.'}
    if not args.foul_only:
        sm, shash, simple = load(SIMPLE)
        if sm['paired_foul_manifest_sha256'] != digest:
            raise ValueError('Simple panel references a different Foul Play panel')
        for row in simple.values():
            reference = foul[row['paired_foul_job_id']]
            for field in ('teams','engine_seed','orientation','root','request_id'):
                if row[field] != reference[field]:
                    raise ValueError(f'Simple comparison differs in {field}')
        simple_roots = per_root(simple)
        comparison = {}
        for request in sm['requests']:
            pair = request['id']
            common = sorted(roots[pair].keys() & simple_roots[pair].keys() & set(sm['root_seeds']))
            if common:
                comparison[pair] = {'roots':common,
                                    'foul_play':float(np.mean([roots[pair][i] for i in common])),
                                    'simple_heuristics':float(np.mean([simple_roots[pair][i] for i in common]))}
        summary['pilot_comparison'] = {
            **paired_summary([(r['foul_play'],r['simple_heuristics']) for r in comparison.values()]),
            'simple_manifest_sha256': shash, 'simple_battles': len(simple),
            'simple_statuses': dict(Counter(r['status'] for r in simple.values())), 'per_pair':comparison}
    dump(FOUL/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'per_pair'}, indent=2))


if __name__ == '__main__':
    main()
