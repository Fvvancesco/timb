"""Compare aggregate and explicit set features on the same simulated development folds."""
from collections import Counter
import argparse
from functools import lru_cache
import json
from pathlib import Path
from statistics import mean

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_supervised import ROOT, SEEDS, V1
from icon_pokemon.adapter import save_json, sha256, showdown
from icon_pokemon.features import STATS, TeamFeatures
from icon_pokemon.knowledge import build_graph, capability_evidence
from icon_pokemon.supervised import (fit_model, fold_members, load_examples, metrics,
                                     partition, probabilities)

ABLATIONS = {
    'aggregate': frozenset(),
    'moves': {'move'},
    'items': {'item'},
    'abilities': {'ability'},
    'moves_items_abilities': {'move', 'item', 'ability'},
    'complete_without_id': {'species', 'move', 'item', 'ability', 'nature',
                            'teraType', 'gender', 'hpType', 'level', 'ev', 'iv'},
    'explicit_sets': None,
}


def explicit_sets(ids, records):
    """Keep set components tied to their bearer, independently of team order."""
    encoded = Counter()
    for identifier in ids:
        record = records[identifier]
        details = record['set']
        species = record['species_id']
        encoded[f'species/{species}'] += 1
        for field in ('item', 'ability', 'nature', 'teraType', 'gender', 'hpType'):
            value = details.get(field) or '<absent>'
            encoded[f'{species}/{field}/{value}'] += 1
        for move in record['moves']:
            encoded[f'{species}/move/{move["id"]}'] += 1
        encoded[f'{species}/level'] += details.get('level', 100) / 100
        for stat in STATS:
            encoded[f'{species}/ev/{stat}'] += details.get('evs', {}).get(stat, 0) / 252
            encoded[f'{species}/iv/{stat}'] += details.get('ivs', {}).get(stat, 31) / 31
        encoded[f'set/{identifier}'] += 1
    return dict(encoded)


def augmented_logistic(x, y):
    middle = len(x[0]) // 2
    swapped = [row[middle:] + row[:middle] for row in x]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    return model.fit(x + swapped, y + [1 - value for value in y],
                     logisticregression__sample_weight=[0.5] * (2 * len(y)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', type=Path, default=[
        V1 / 'initial_train_labels.jsonl', V1 / 'distributed_train_labels.jsonl'])
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/complete_sets_probe.json')
    args = parser.parse_args()
    catalog_path = V1 / 'catalog.json'
    label_paths = args.labels
    catalog = json.loads(catalog_path.read_text())
    labels = [json.loads(line) for path in label_paths for line in path.read_text().splitlines() if line]
    teams = {i: team['set_ids'] for i, team in catalog['teams'].items()}
    examples, outcomes = load_examples(labels, catalog['teams'])
    records = {r['id']: r for r in catalog['sets']}
    mechanics = showdown('export-mechanics.cjs', {'format': catalog['format'],
        'move_ids': sorted({m['id'] for r in records.values() for m in r['moves']})})
    _, graph = build_graph(catalog, mechanics)
    evidence = {i: capability_evidence(graph, record) for i, record in records.items()}
    features = TeamFeatures(records, graph, mechanics, evidence)

    @lru_cache(maxsize=None)
    def aggregate(a, b):
        return features.pair(teams[a], teams[b])

    observed = {i for a, b, _ in examples for i in (a, b)}
    explicit = {i: explicit_sets(teams[i], records) for i in observed}

    def component(key):
        first, second, *_ = key.split('/')
        return first if first in ('set', 'species') else second

    def features_for_variant(groups):
        if groups is None:
            return explicit
        return {team: {key: value for key, value in profile.items()
                       if component(key) in groups}
                for team, profile in explicit.items()}

    ablations = {name: features_for_variant(groups) for name, groups in ABLATIONS.items()}

    def compare(train, test, discarded):
        train_teams = {i for a, b, _ in train for i in (a, b)}
        test_teams = {i for a, b, _ in test for i in (a, b)}
        assert not train_teams & test_teams
        summary = {'train': len(train), 'test': len(test), 'discarded': discarded}
        if len({y for _, _, y in train}) < 2 or not test:
            return summary

        ids = sorted(train_teams | test_teams)
        def vector(a, b, team_vectors):
            values = aggregate(a, b)
            if team_vectors is None:
                return values
            mid = len(values) // 2
            return (*values[:mid], *team_vectors[a], *values[mid:], *team_vectors[b])

        y_train, y_test = [y for _, _, y in train], [y for _, _, y in test]
        summary['matches'] = [{'team_a': a, 'team_b': b, 'outcome': y} for a, b, y in test]
        summary['metrics'] = {'neutral': metrics(y_test, [0.5] * len(test))}
        summary['extra_features_per_side'] = {}
        for name, profiles in ablations.items():
            team_vectors = None
            if any(profiles[i] for i in train_teams):
                vectorizer = DictVectorizer(sparse=False)
                vectorizer.fit(profiles[i] for i in sorted(train_teams))
                encoded = vectorizer.transform([profiles[i] for i in ids])
                team_vectors = dict(zip(ids, encoded, strict=True))
                summary['extra_features_per_side'][name] = len(vectorizer.feature_names_)
            x_train = [vector(a, b, team_vectors) for a, b, _ in train]
            x_test = [vector(a, b, team_vectors) for a, b, _ in test]
            models = {name: fit_model('logistic', 0.1, x_train, y_train),
                      name + '_augmented': augmented_logistic(x_train, y_train)}
            for variant, model in models.items():
                prediction = probabilities(model, x_test)
                summary['metrics'][variant] = metrics(y_test, prediction)
                for row, p in zip(summary['matches'], prediction, strict=True):
                    row[variant] = float(p)
        train_sets = {i for team in train_teams for i in teams[team]}
        test_sets = {i for team in test_teams for i in teams[team]}
        summary['unseen_exact_test_sets'] = len(test_sets - train_sets)
        summary['distinct_test_sets'] = len(test_sets)
        return summary

    results = {}
    for seed in SEEDS:
        results[str(seed)] = [
            compare(*partition(examples, observed - held, held))
            for held in fold_members(observed, seed)
        ]
    usable = [fold for folds in results.values() for fold in folds if 'metrics' in fold]
    evaluated_pairs = {(row['team_a'], row['team_b']) for fold in usable for row in fold['matches']}
    report = {'purpose': 'exploratory development ablation, logistic C=0.1 fixed',
              'augmentation': 'For training only, swap team feature blocks and use 1-y; weight each orientation 0.5.',
              'outcomes': outcomes, 'decisive_matches': len(examples),
              'distinct_unordered_pairs': len({tuple(sorted((a, b))) for a, b, _ in examples}),
              'matches_evaluated_at_least_once': sum((a, b) in evaluated_pairs for a, b, _ in examples),
              'usable_folds': len(usable), 'folds': results,
              'mean_fold_log_loss': {variant: mean(fold['metrics'][variant]['log_loss'] for fold in usable)
                                     for variant in usable[0]['metrics']},
              'input_hashes': {str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT)
                               else str(path.resolve()): sha256(path)
                               for path in [catalog_path, *label_paths]}}
    save_json(args.output, report)
    print(args.output)


if __name__ == '__main__':
    main()
