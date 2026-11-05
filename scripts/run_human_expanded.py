"""Compare preview and rating features on human replays with a separate holdout."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from math import prod
from pathlib import Path
from statistics import mean

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from run_human_preview import ROOT, read_preview
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.features import STATS
from icon_pokemon.supervised import fold_members, metrics, partition, probabilities

C_VALUES = (0.01, 0.1, 1.0, 10.0)
DEPTHS = (1, 2, 3)
FEATURES = ('rating', 'preview', 'preview_species', 'preview_rating', 'preview_species_rating')


def source_rows(path):
    if path.suffix == '.parquet':
        from pyarrow.parquet import ParquetFile
        parquet = ParquetFile(path)
        columns = (['id', 'formatid', 'log'] if 'formatid' in parquet.schema_arrow.names
                   else ['battle_id', 'gamemode', 'elo', 'text'])
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                if 'elo' not in row or row['elo'] in ('1600-1799', '1800+'):
                    yield row
    else:
        data = json.loads(path.read_text())
        yield from data if isinstance(data, list) else (entry['row'] for entry in data['rows'])


def ingest(paths, min_rating):
    accepted, counts = {}, {}
    for path in paths:
        for row in source_rows(path):
            source = 'metamon' if 'formatid' in row else 'pokechamp'
            counts.setdefault(source, Counter())
            status, example = read_preview(row, min_rating)
            counts[source][status] += 1
            if example is None:
                continue
            if example['id'] in accepted:
                counts[source]['duplicate'] += 1
                continue
            accepted[example['id']] = (source, example)

    names = sorted({name for _, row in accepted.values()
                    for side in ('team_a', 'team_b') for name in row[side]})
    species = showdown('export-preview-species.cjs', {'names': names})
    prepared = {}
    for identifier, (source, row) in accepted.items():
        if any(n not in species for side in ('team_a', 'team_b') for n in row[side]):
            counts[source]['unknown_or_masked_form'] += 1
            continue
        sides = [tuple(sorted(row[k])) for k in ('team_a', 'team_b')]
        groups = ['|'.join(sorted(species[n]['base_species'] for n in side)) for side in sides]
        prepared[identifier] = {**row, 'source': source, 'sides': sides, 'groups': groups}
        counts[source]['usable'] += 1
    return prepared, species, counts


def rating_from_example(row):
    return (row['ratings'][0] - row['ratings'][1]) / 400


def model_input(variant, row, profiles, species):
    a, b = row['sides']
    if variant == 'boosting_preview':
        return (*profiles(a).values(), *profiles(b).values())
    result = {}
    if variant != 'rating':
        result.update({k: value - profiles(b)[k] for k, value in profiles(a).items()})
    if 'species' in variant:
        occurrences = Counter(f'species_{species[n]["id"]}' for n in a)
        occurrences.subtract(f'species_{species[n]["id"]}' for n in b)
        result.update(occurrences)
    if 'rating' in variant:
        result['rating_gap'] = rating_from_example(row)
    return result


def evaluate(prepared, species, previous_ids, chart):
    @lru_cache(maxsize=None)
    def profiles(side):
        entries = [species[n] for n in side]
        values = {}
        for kind, interactions in chart.items():
            factors = [prod(interactions[t] for t in s['types']) for s in entries]
            values[f'{kind}_member'] = sum(kind in s['types'] for s in entries) / 6
            values[f'{kind}_weak'] = sum(m > 1 for m in factors) / 6
            values[f'{kind}_resist'] = sum(0 < m < 1 for m in factors) / 6
            values[f'{kind}_immune'] = sum(m == 0 for m in factors) / 6
        for stat in STATS:
            column = [entry['stats'][stat] for entry in entries]
            for label, aggregate in (('mean', mean), ('min', min), ('max', max)):
                values[f'{stat}_{label}'] = aggregate(column) / 100
        return values

    rows = [(*row['groups'], identifier) for identifier, row in prepared.items()]
    groups = {i for a, b, _ in rows for i in (a, b)}
    held = fold_members(groups, 20261103)[0]
    development = groups - held
    train_rows, held_rows, dropped = partition(rows, development, held)
    folds = [partition(train_rows, development - valid, valid)
             for valid in fold_members(development, 11)]
    train_ids = [i for _, _, i in train_rows]
    test_ids = [i for _, _, i in held_rows if i not in previous_ids]
    if not test_ids or any({prepared[i]['y'] for _, _, i in side} != {0, 1}
                           for fit, valid, _ in folds for side in (fit, valid)):
        raise ValueError('Insufficient examples for the preselected grouped split')

    def labels(ids):
        return [prepared[i]['y'] for i in ids]

    ranked = {}
    fitted = {}
    for variant in (*FEATURES, 'boosting_preview'):
        x = {i: model_input(variant, row, profiles, species) for i, row in prepared.items()}
        boosting = variant == 'boosting_preview'

        def fit(ids, option):
            if boosting:
                model = HistGradientBoostingClassifier(max_depth=option, max_iter=80,
                                                       early_stopping=False, random_state=11)
            else:
                model = make_pipeline(DictVectorizer(), LogisticRegression(
                    C=option, fit_intercept=False, max_iter=1000))
            return model.fit([x[i] for i in ids], labels(ids))

        def predict(model, ids):
            features = [x[i] for i in ids]
            if boosting:
                return probabilities(model, features)
            return model.predict_proba(features)[:, 1]

        losses = {}
        for option in (DEPTHS if boosting else C_VALUES):
            scores = []
            for fit_rows, valid_rows, _ in folds:
                fit_ids = [i for _, _, i in fit_rows]
                valid_ids = [i for _, _, i in valid_rows]
                scores.append(metrics(labels(valid_ids), predict(fit(fit_ids, option), valid_ids))['log_loss'])
            losses[option] = mean(scores)
        chosen = min(losses, key=losses.__getitem__)
        ranked[variant] = {'parameter': 'max_depth' if boosting else 'C',
                           'chosen': chosen, 'cv_log_loss': losses[chosen], 'grid': losses}
        model = fit(train_ids, chosen)
        fitted[variant] = predict(model, test_ids)

    model_variants = [name for name in ranked if name != 'rating']
    winner = min(model_variants, key=lambda name: ranked[name]['cv_log_loss'])
    fitted['neutral'] = [0.5] * len(test_ids)
    fitted['train_rate'] = [mean(labels(train_ids))] * len(test_ids)
    sources = {prepared[i]['source'] for i in test_ids}
    by_source = {}
    for source in ['all', *sorted(sources)]:
        positions = [j for j, i in enumerate(test_ids)
                     if source == 'all' or prepared[i]['source'] == source]
        subset = labels([test_ids[j] for j in positions])
        by_source[source] = {'matches': len(positions),
                             'metrics': {name: metrics(subset, [p[j] for j in positions])
                                         for name, p in fitted.items()}}
    return {'selected_on_development': winner, 'grid_search': ranked, 'held_out': by_source,
            'development_matches': len(train_ids), 'crossing_matches_removed': dropped,
            'prior_seen_matches_removed_from_holdout': len(held_rows) - len(test_ids),
            'development_team_groups': len(development), 'held_out_team_groups': len(held),
            'development_source_counts': dict(Counter(prepared[i]['source'] for i in train_ids)),
            'held_out_battle_ids': test_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--previous', nargs='+', required=True, type=Path)
    parser.add_argument('--min-rating', type=int, default=1600)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/human_expanded.json')
    args = parser.parse_args()
    previous_ids = set()
    for path in args.previous:
        for row in source_rows(path):
            _, parsed = read_preview(row, args.min_rating)
            if parsed:
                previous_ids.add(parsed['id'])
    prepared, species, counts = ingest(args.inputs, args.min_rating)
    mechanics = showdown('export-mechanics.cjs', {'format': 'gen9ou'})
    report = {'purpose': 'exploratory human replay classification', 'counts': counts,
              'min_rating_both': args.min_rating,
              'input_hashes': {str(p): sha256(p) for p in [*args.inputs, *args.previous]}}
    report.update(evaluate(prepared, species, previous_ids, mechanics['chart']))
    save_json(args.output, report)
    print(json.dumps({k: report[k] for k in ('counts', 'development_matches', 'selected_on_development')}))


if __name__ == '__main__':
    main()
