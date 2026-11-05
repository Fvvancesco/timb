"""Create and evaluate a composition-disjoint chronological Metamon split."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from run_team_completion_extension import ROOT, evaluate, freeze
from run_human_preview import read_preview
from icon_pokemon.adapter import sha256, showdown

OUT = ROOT / 'data/processed/team_completion_temporal_v1'
REPORT = ROOT / 'reports/team_completion_temporal_v1'
PRIOR = ROOT / 'data/processed/team_completion_v1'
REVISION = '060367c26630e0d36308fd42673e5cf2f4b4421a'


def period(year, month):
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def prepare(training, tests):
    from pyarrow.parquet import ParquetFile

    mar, apr, jun = (period(2026, month) for month in (3, 4, 6))
    accepted, battle_ids, counts = {}, set(), Counter()
    for path, is_test in [(training, False), *((test, True) for test in tests)]:
        for batch in ParquetFile(path).iter_batches(batch_size=1024, columns=['id', 'formatid', 'uploadtime', 'log']):
            for row in batch.to_pylist():
                stamp = int(row.get('uploadtime') or 0)
                if is_test and not apr <= stamp < jun:
                    continue
                if not is_test and not 0 < stamp < apr:
                    continue
                split = 'test' if is_test else 'train' if stamp < mar else 'validation'
                status, item = read_preview(row, 1600)
                counts[f'{split}/{status}'] += 1
                if item is None:
                    continue
                if item['id'] in battle_ids:
                    counts[f'{split}/duplicate_battle'] += 1
                    continue
                battle_ids.add(item['id'])
                accepted[item['id']] = split, item
    names = {n for _, r in accepted.values() for k in ('team_a', 'team_b') for n in r[k]}
    species = showdown('export-preview-species.cjs', {'names': sorted(names)})
    groups = {k: set() for k in ('train','validation','test')}
    for split, row in accepted.values():
        for side in ('team_a','team_b'):
            if all(n in species for n in row[side]):
                team = tuple(sorted(species[n]['base_species'] for n in row[side]))
                if len(set(team)) == 6:
                    groups[split].add(team)
    previous = set()
    old_files = [PRIOR / 'splits.json', PRIOR / 'external_metamon31.json', PRIOR / 'external_metamon32.json']
    for p in old_files:
        saved = json.loads(p.read_text())
        past = saved['teams'] if 'teams' in saved else [t for teams in saved['splits'].values() for t in teams]
        previous.update(tuple(sorted(t)) for t in past)
    catalog = ROOT / 'data/processed/dataset_v1/catalog.json'
    previous.update(tuple(sorted(t['species_composition'])) for t in json.loads(catalog.read_text())['teams'].values())
    before = {k: len(v) for k, v in groups.items()}
    groups['validation'] -= groups['train']
    groups['test'] -= groups['train'] | groups['validation'] | previous
    after = {k: len(v) for k, v in groups.items()}
    if min(after.values()) < 100:
        raise ValueError(f'Too few eligible unique teams. Before {before}, after {after}')
    manifest = {'source_sha256': {p.name: sha256(p) for p in (training, *tests, catalog, *old_files, Path(__file__))},
                'source_revision': REVISION, 'source_counts': counts, 'minimum_rating': 1600,
                'format': 'gen9ou', 'temporal_field': 'Metamon uploadtime UTC',
                'periods': {'train': 'before 2026-03-01', 'validation': 'March 2026', 'test': 'April–May 2026'},
                'source_shards': {'train': 30, 'validation': 30, 'test': [p.name for p in tests]},
                'policy': 'Distinct compositions; test excludes all earlier project completion splits, external holds and solver catalog. Training and model selection use past dates only. The May-only window had 73 eligible teams and was expanded before any model was trained or evaluated.',
                'compositions_before_filter': before, 'compositions_after_filter': after,
                'splits': {name: [list(t) for t in sorted(teams)] for name, teams in groups.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'splits.json', manifest)
    print(before, after, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_shard', type=Path)
    parser.add_argument('test_shards', type=Path, nargs='+')
    args = parser.parse_args()
    prepare(args.train_shard, args.test_shards)
    evaluate(OUT / 'splits.json', REPORT)


if __name__ == '__main__':
    main()
