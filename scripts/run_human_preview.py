"""Check the supervised pipeline on human Gen9 OU replays using Team Preview."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from math import prod
from pathlib import Path
import re
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.features import STATS
from icon_pokemon.supervised import (evaluate_nested, fit_model, fold_counts, fold_members,
                                     metrics, partition, probabilities)


def read_preview(row, min_rating):
    log = row.get('log', row.get('text', ''))
    format_id = row.get('formatid', row.get('gamemode'))
    if format_id != 'gen9ou':
        return 'format', None
    players, ratings = {}, {}
    roster = {'p1': [], 'p2': []}
    winner, started, preview = None, False, False
    for line in log.splitlines():
        fields = line.split('|')
        if len(fields) < 2:
            continue
        event = fields[1].lower()
        if event in ('start', 'turn'):
            started = True
        elif event == 'win' and len(fields) >= 3:
            winner = fields[2].strip().casefold()
        elif not started:
            if event == 'teampreview':
                preview = True
            elif event == 'clearpoke':
                roster = {'p1': [], 'p2': []}
            elif event == 'player' and len(fields) >= 6 and fields[2] in roster:
                players[fields[2]] = fields[3].strip().casefold()
                ratings[fields[2]] = int(fields[5]) if fields[5].isdigit() else 0
            elif event == 'poke' and len(fields) >= 4 and fields[2] in roster:
                roster[fields[2]].append(fields[3].split(',')[0].strip())
    if not preview or not started or any(len(team) != 6 for team in roster.values()):
        return 'incomplete_preview', None
    if any(ratings.get(side, 0) < min_rating for side in roster):
        return 'rating', None
    winners = [side for side in roster if players.get(side) == winner]
    if winner is None or len(winners) != 1:
        return 'non_decisive_or_unresolved', None
    battle = re.match(r'^(?:gen9ou-)?(\d+)', row.get('id', row.get('battle_id', '')))
    if not battle:
        return 'missing_battle_id', None
    return 'parsed', {'id': battle[1], 'team_a': roster['p1'], 'team_b': roster['p2'],
                      'ratings': [ratings['p1'], ratings['p2']], 'y': int(winners[0] == 'p1')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--min-rating', type=int, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/human_preview.json')
    args = parser.parse_args()
    counts, accepted = Counter(), {}
    for path in args.inputs:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else [entry['row'] for entry in data['rows']]
        for row in rows:
            status, example = read_preview(row, args.min_rating)
            counts[status] += 1
            if example:
                if example['id'] in accepted:
                    counts['duplicate_battle'] += 1
                else:
                    accepted[example['id']] = example
    names = sorted({name for row in accepted.values() for side in ('team_a', 'team_b') for name in row[side]})
    species = showdown('export-preview-species.cjs', {'names': names})
    mechanics = showdown('export-mechanics.cjs', {'format': 'gen9ou'})
    previews = {}
    examples = []
    for row in accepted.values():
        if any(name not in species for side in ('team_a', 'team_b') for name in row[side]):
            counts['ambiguous_species'] += 1
            continue
        keys = []
        for side in ('team_a', 'team_b'):
            entries = sorted((species[name] for name in row[side]), key=lambda s: s['id'])
            key = '|'.join(sorted(s['base_species'] for s in entries))
            previews.setdefault(key, {})[tuple(s['id'] for s in entries)] = entries
            keys.append(key)
        examples.append((tuple(keys), row['y'], row))
    # Group alternate forms of a composition together but use each actual preview for its feature vector.
    profiles = {}
    prepared = []
    for (a, b), y, row in examples:
        ids = []
        for side, key in [('team_a', a), ('team_b', b)]:
            ident = tuple(sorted(species[name]['id'] for name in row[side]))
            profiles[key, ident] = previews[key][ident]
            ids.append(ident)
        prepared.append((a, b, y, tuple(ids)))
    # The current CV API keys vectors by grouped team. Use one deterministic preview per
    # composition, retaining only matches whose previews exactly match those representatives.
    reps = {key: min(variants) for key, variants in previews.items()}
    rows = [(a, b, y) for a, b, y, ids in prepared if ids == (reps[a], reps[b])]
    counts['alternate_preview_removed'] = len(prepared) - len(rows)

    @lru_cache(maxsize=None)
    def vector_side(key):
        entries = profiles[key, reps[key]]
        values = []
        for kind in mechanics['types']:
            multipliers = [prod(mechanics['chart'][kind][t] for t in entry['types']) for entry in entries]
            values.extend((sum(kind in s['types'] for s in entries), sum(m > 1 for m in multipliers),
                           sum(0 < m < 1 for m in multipliers), sum(m == 0 for m in multipliers)))
        for stat in STATS:
            column = [entry['stats'][stat] for entry in entries]
            values.extend((mean(column), min(column), max(column)))
        return tuple(values)

    def vector(a, b):
        return (*vector_side(a), *vector_side(b))

    teams = {i for a, b, _ in rows for i in (a, b)}
    report = {'purpose': 'human replay development only', 'min_both_player_rating': args.min_rating,
              'counts': dict(counts), 'matches': len(rows), 'compositions': len(teams),
              'input_hashes': {str(p): sha256(p) for p in args.inputs}}
    if len(teams) >= 6:
        report['nested'] = evaluate_nested(rows, teams, vector)
        if report['nested']['status'] != 'ok':
            test_teams = fold_members(teams, 11)[0]
            train, test, dropped = partition(rows, teams - test_teams, test_teams)
            report['smoke_fold'] = fold_counts(train, test, dropped)
            if len({y for _, _, y in train}) == 2 and test:
                train_x, train_y = [vector(a, b) for a, b, _ in train], [y for _, _, y in train]
                test_x, test_y = [vector(a, b) for a, b, _ in test], [y for _, _, y in test]
                report['smoke_metrics'] = {'neutral': metrics(test_y, [0.5] * len(test))}
                for family, option in [('logistic', 0.1), ('boosting', 2)]:
                    model = fit_model(family, option, train_x, train_y)
                    report['smoke_metrics'][family] = metrics(test_y, probabilities(model, test_x))
    save_json(args.output, report)
    print(json.dumps({k: report[k] for k in ('matches', 'compositions', 'counts')}))


if __name__ == '__main__':
    main()
