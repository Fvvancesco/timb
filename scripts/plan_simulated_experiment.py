"""Freeze the v2 battle design from the v1 catalog without reading battle outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256
from icon_pokemon.dataset import make_jobs
from icon_pokemon.supervised import nested_folds

CATALOG = Path('data/processed/dataset_v1/catalog.json')
V1 = Path('config/dataset_v1_manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'config/dataset_v2_manifest.json')
    args = parser.parse_args()
    catalog = json.loads((ROOT / CATALOG).read_text())
    prior = json.loads((ROOT / V1).read_text())
    if sha256(ROOT / CATALOG) != prior['hashes'][str(CATALOG)]:
        raise ValueError('Expected the verified v1 catalog')
    groups = {split: [t for t in catalog['teams'].values() if t['split'] == split]
              for split in ('train', 'validation', 'test')}
    if tuple(map(len, groups.values())) != (100, 20, 20):
        raise ValueError('Unexpected split sizes')
    if len({tuple(sorted(t['species_composition'])) for teams in groups.values() for t in teams}) != 140:
        raise ValueError('Duplicate composition in catalog')
    counts = (280, 40, 40)
    jobs = make_jobs(groups, counts, roots=(101,))
    repetitions = make_jobs(groups, counts, roots=(103,))
    repeated_pairs = sorted({j['pair_id'] for j in repetitions if j['split'] == 'train'})[:20]
    jobs.extend(j for j in repetitions if j['split'] == 'train' and j['pair_id'] in repeated_pairs)
    split_order = {split: i for i, split in enumerate(groups)}
    jobs.sort(key=lambda job: (split_order[job['split']], job['pair_id'],
                               job['root_seed'], job['orientation']))
    if {j['job_id'] for j in jobs} & {j['job_id'] for j in prior['jobs']}:
        raise ValueError('New jobs overlap v1 job IDs')
    development = {t['id'] for split in ('train', 'validation') for t in groups[split]}
    # Count unique pairs rather than sides or seed repetitions. The third field is unused here.
    pairs = sorted({(*sorted((j['team_a_id'], j['team_b_id'])), 0)
                    for j in jobs if j['split'] != 'test'})
    fold_counts = {
        str(seed): [{'outer_train_pairs': len(train), 'outer_test_pairs': len(test),
                     'outer_crossing_pairs': dropped,
                     'inner_train_pairs': [len(fit) for fit, _, _ in inner],
                     'inner_test_pairs': [len(valid) for _, valid, _ in inner]}
                    for train, test, dropped, inner in nested_folds(pairs, development, seed)]
        for seed in (11, 23, 47)
    }
    keys = ('format', 'showdown_version', 'pilot', 'pilot_commit', 'search_time_ms',
            'search_parallelism', 'search_threads', 'max_turns', 'wall_limit_seconds',
            'source_kind', 'label', 'catalog_path')
    manifest = {key: prior[key] for key in keys}
    manifest.update(dataset_version='v2', planned_max_jobs=len(jobs),
                    development_jobs=sum(j['split'] != 'test' for j in jobs),
                    job_counts=dict(Counter(j['split'] for j in jobs)),
                    split_sizes={k: len(v) for k, v in groups.items()},
                    fold_counts_before_outcomes=fold_counts,
                    seed_policy='Keep each consecutive pair group on the same machine. Same engine seed for both sides within each root. Native timed search remains nondeterministic.',
                    pilot_censor_policy='Continue only on recognized zero-visit Foul Play crash. Preserve logs, assign null outcome and do not retry. Other errors stop the batch.',
                    test_policy='Hold out test outcomes until model and candidate selection protocol are fixed.',
                    hashes={str(CATALOG): sha256(ROOT / CATALOG), str(V1): sha256(ROOT / V1)},
                    jobs=jobs)
    if args.output.exists():
        if json.loads(args.output.read_text()) != manifest:
            raise ValueError(f'Refusing to overwrite different plan at {args.output}')
    else:
        save_json(args.output, manifest)
    print(json.dumps(manifest['job_counts']))


if __name__ == '__main__':
    main()
