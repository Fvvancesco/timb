"""Generate frozen additional development pairs, excluding previous manifests."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256
from icon_pokemon.dataset import make_jobs, stable_id

CONFIG = ROOT / 'config'
CATALOG = Path('data/processed/dataset_v1/catalog.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('v3', 'v4', 'v5'), default='v3')
    args = parser.parse_args()
    versions = tuple(f'v{i}' for i in range(1, int(args.dataset[1:])))
    previous = [CONFIG / f'dataset_{version}_manifest.json' for version in versions]
    manifests = [json.loads(path.read_text()) for path in previous]
    v2 = manifests[1]
    if sha256(ROOT / CATALOG) != v2['hashes'][str(CATALOG)]:
        raise ValueError('Catalog differs from the frozen catalog')
    forbidden = {j['pair_id'] for manifest in manifests for j in manifest['jobs']}
    catalog = json.loads((ROOT / CATALOG).read_text())
    teams = {split: [team for team in catalog['teams'].values() if team['split'] == split]
             for split in ('train', 'validation')}
    rng = Random({'v3': 20261104, 'v4': 20261105, 'v5': 20261106}[args.dataset])
    counts = {'v3': (280, 40), 'v4': (2000, 100), 'v5': (1000, 10)}[args.dataset]
    pairs = {}
    for (split, members), count in zip(teams.items(), counts, strict=True):
        candidates = [pair for pair in combinations(sorted(t['id'] for t in members), 2)
                      if stable_id(pair) not in forbidden]
        pairs[split] = sorted(rng.sample(candidates, count))
    root_seed = {'v3': 107, 'v4': 211, 'v5': 307}[args.dataset]
    jobs = make_jobs(teams, roots=(root_seed,), explicit_pairs=pairs)
    order = {split: i for i, split in enumerate(teams)}
    jobs.sort(key=lambda j: (order[j['split']], j['pair_id'], j['orientation']))
    assert len({j['pair_id'] for j in jobs}) == sum(counts)
    assert not forbidden.intersection(j['pair_id'] for j in jobs)
    manifest = {k: v for k, v in v2.items()
                if k not in ('jobs', 'fold_counts_before_outcomes', 'split_sizes')}
    manifest.update(dataset_version=args.dataset, planned_max_jobs=len(jobs), development_jobs=len(jobs),
                    job_counts=dict(Counter(j['split'] for j in jobs)),
                    split_sizes={split: len(members) for split, members in teams.items()},
                    sampling=('New pairs from the same development roster, excluding every pair in v1 and v2.'
                              if args.dataset == 'v3' else
                              'New pairs from the same development roster, excluding every pair in v1, v2 and v3. 2000 train and 100 validation pairs, both orientations; seed root 211.'),
                    hashes={str(CATALOG): sha256(ROOT / CATALOG),
                            **{str(p.relative_to(ROOT)): sha256(p) for p in previous}},
                    jobs=jobs)
    if args.dataset == 'v5':
        manifest.update(
            sampling='1000 new train pairs and the 10 remaining validation pairs, excluding all pairs in v1-v4; both orientations; root seed 307.',
            scope='User-authorized additional development collection after core final evaluation. Does not change the frozen core model or supply a new independent final test.',
            planning_code_sha256=sha256(Path(__file__)))
    path = CONFIG / f'dataset_{args.dataset}_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Existing {args.dataset} manifest differs')
    else:
        save_json(path, manifest)
    local = sum(int(j['pair_id'][:8], 16) % 10 < 3 for j in jobs)
    print(f'{len(jobs)} games, {local} local and {len(jobs) - local} remote')


if __name__ == '__main__':
    main()
