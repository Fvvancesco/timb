"""Freeze a seed-replication panel using development pairings, without reading outcomes."""
import json
from itertools import combinations
from pathlib import Path
from random import Random
import sys

from run_coverage_confirmation import ROOT, dump
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256
from icon_pokemon.dataset import engine_seed, stable_id


def main():
    out = ROOT / 'data/processed/matchup_panel_v1'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('Panel is already frozen')
    source = ROOT / 'data/processed/dataset_v1/catalog.json'
    catalog = json.loads(source.read_text())
    teams = {k: v for k, v in catalog['teams'].items() if v['split'] in ('train', 'validation')}
    candidates = list(combinations(sorted(teams), 2))
    Random(1105).shuffle(candidates)
    roots = (811, 823, 829, 839, 853, 857, 859, 863)
    used, requests, jobs = set(), [], []
    for a, b in candidates:
        if a in used or b in used or teams[a]['split'] != teams[b]['split']:
            continue
        used.update((a, b))
        request_id = f'panel-v1-{len(requests):02}'
        requests.append({'id': request_id, 'team_a': a, 'team_b': b, 'split': teams[a]['split']})
        for root in roots:
            for side in (0, 1):
                pair = (a, b) if side == 0 else (b, a)
                key = f'{request_id}-foul-{root}-{side}'
                token = stable_id([key])[:10]
                jobs.append({'job_id': key, 'request_id': request_id, 'methods': ['foul_play'],
                             'root': root, 'orientation': side, 'players': ['IP'+token+'a', 'IP'+token+'b'],
                             'engine_seed': engine_seed(request_id, root),
                             'python_seeds': [int(stable_id([request_id,str(root),str(side),str(i)])[:8],16)
                                              for i in (0, 1)],
                             'teams': [{**{k: teams[i][k] for k in ('sets','packed')},
                                        'text': teams[i]['text'].strip()+'\n'} for i in pair]})
        if len(requests) == 30:
            break
    assert len(requests) == 30 and len(used) == 60 and len(jobs) == 480
    runner = Path(__file__).with_name('run_coverage_confirmation.py')
    dump(out / 'manifest.json', {
        'format': catalog['format'], 'source_sha256': {p.name: sha256(p) for p in (source, runner, Path(__file__))},
        'requests': requests, 'jobs': jobs, 'root_seeds': roots,
        'root_groups': [roots[:4], roots[4:]],
        'protocol': '30 team-disjoint pairs sampled from the development roster without outcomes. Foul Play in self-play. Average side orientations within each root. Compare disjoint groups of four roots per pair.',
        'second_pilot_subset': {'requests': [r['id'] for r in requests[:15]], 'roots': roots[:4], 'both_sides': True},
        'missing_policy': 'Use only roots with both orientations available for split-half analysis. Report eligible pairs and all excluded outcomes. No test squads.',
    })
    print('Frozen 30 disjoint pairs, 60 development teams and 480 battles')


if __name__ == '__main__':
    main()
