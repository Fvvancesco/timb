"""Merge a complete two-host development collection without fitting any model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('v2', 'v3', 'v4', 'v5'), default='v5')
    args = parser.parse_args()
    manifest_path = ROOT / f'config/dataset_{args.dataset}_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    catalog_path = ROOT / manifest['catalog_path']
    if sha256(catalog_path) != manifest['hashes'][manifest['catalog_path']]:
        raise ValueError('Catalog hash differs')
    teams = json.loads(catalog_path.read_text())['teams']
    expected = {j['job_id']: j for j in manifest['jobs'] if j['split'] in ('train', 'validation')}
    state = ROOT / f'data/processed/dataset_{args.dataset}'
    sources = [state / shard / 'labels.jsonl' for shard in ('local', 'remote')]
    collected, counts = {}, {}
    for source in sources:
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        counts[source.parent.name] = len(rows)
        for row in rows:
            identifier = row['job_id']
            if identifier in collected or identifier not in expected:
                raise ValueError('Duplicate or unexpected job')
            if any(row.get(k) != value for k, value in expected[identifier].items()):
                raise ValueError('Job metadata differs from the frozen manifest')
            if any(teams[row[k]]['split'] != row['split'] for k in ('team_a_id', 'team_b_id')):
                raise ValueError('Team partition differs')
            if row['outcome'] in ('team_a', 'team_b', 'draw') and not row['replay_verified']:
                raise ValueError('Outcome without replay verification')
            collected[identifier] = row
    if set(collected) != set(expected):
        raise ValueError('Collection incomplete; refusing a final merged export')
    output = state / 'development_labels.jsonl'
    if output.exists():
        previous = {r['job_id']: r for r in map(json.loads, output.read_text().splitlines())}
        if any(collected.get(i) != row for i, row in previous.items()):
            raise ValueError('Previously exported outcomes changed')
    temporary = output.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(collected[i], sort_keys=True) + '\n' for i in sorted(collected)))
    temporary.replace(output)
    decisive = sum(r['outcome'] in ('team_a', 'team_b') for r in collected.values())
    report = {'dataset': args.dataset, 'complete': True, 'records': len(collected),
              'decisive': decisive, 'by_host': counts,
              'statuses': dict(Counter(r['status'] for r in collected.values())),
              'outcomes': dict(Counter(str(r['outcome']) for r in collected.values())),
              'scope': 'Additional development data only. Frozen core experiment and model remain unchanged.',
              'hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                         (manifest_path, catalog_path, *sources, output)}}
    save_json(ROOT / f'reports/synthetic_development_{args.dataset}_status.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'hashes'}, indent=2))


if __name__ == '__main__':
    main()
