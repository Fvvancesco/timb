"""Freeze the second-pilot subset already selected by the Foul Play manifest."""
import json
from pathlib import Path
import sys

from run_coverage_confirmation import ROOT, dump, assign
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256


def main():
    source = ROOT / 'data/processed/matchup_panel_v1'
    parent = source / 'manifest.json'
    manifest = json.loads(parent.read_text())
    out = ROOT / 'data/processed/matchup_panel_simple_v1'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('Simple panel is already frozen')
    subset = manifest['second_pilot_subset']
    requests = [r for r in manifest['requests'] if r['id'] in subset['requests']]
    jobs = [{**j, 'job_id': j['job_id'].replace('-foul-', '-simple-'),
             'paired_foul_job_id': j['job_id'], 'methods': ['simple_heuristics']}
            for j in manifest['jobs'] if j['request_id'] in subset['requests'] and j['root'] in subset['roots']]
    assert len(requests) == 15 and len(jobs) == 120
    wrapper = ROOT / 'tools/pilots/simple_packed_wrapper.py'
    dump(out / 'manifest.json', {
        'format': manifest['format'], 'source_sha256': {**manifest['source_sha256'], wrapper.name: sha256(wrapper)},
        'requests': requests, 'jobs': jobs, 'root_seeds': subset['roots'], 'paired_foul_manifest_sha256': sha256(parent),
        'protocol': 'SimpleHeuristicsPlayer poke-env 0.16.1 self-play. Same 15 pairs, both orientations and four engine seeds per pair as Foul Play, with same packed sets and per-client seeds. Same host for each pair; both arms retain censoring.',
    })
    for name, count, workers in [('workstation', 10, 14), ('laptop', 5, 6)]:
        assign(out, name, count, workers)
        original = json.loads((source / 'assignments' / f'{name}.json').read_text())
        new = json.loads((out / 'assignments' / f'{name}.json').read_text())
        assert set(new['requests']) <= set(original['requests'])


if __name__ == '__main__':
    main()
