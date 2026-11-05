"""Audit saved extension artifacts without fitting models or changing results."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import save_json, sha256

BASE = 'reports/team_completion_v1/'
DATA = 'data/processed/team_completion_v1/'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads((ROOT / path).read_text())


def verify_hashes(hashes):
    for path, digest in hashes.items():
        require(sha256(ROOT / path) == digest, f'Changed artifact: {path}')


def compositions(teams):
    keys = [tuple(sorted(t)) for t in teams]
    require(all(len(t) == len(set(t)) == 6 for t in keys), 'Invalid composition')
    require(len(keys) == len(set(keys)), 'Repeated unordered composition')
    return set(keys)


def audit():
    splits = read(DATA + 'splits.json')['splits']
    groups = {name: compositions(teams) for name, teams in splits.items()}
    require(all(not a & b for a, b in combinations(groups.values(), 2)), 'Split overlap')
    selection = read(BASE + 'selection.json')
    verify_hashes({DATA + 'splits.json': selection['splits_sha256'],
                   'scripts/run_team_completion_extension.py': selection['code_sha256']})
    vocabulary = set(selection['candidate_species'])
    require(sorted(vocabulary) == selection['candidate_species'], 'Invalid vocabulary')
    require(vocabulary == {s for t in splits['train'] for s in t}, 'Non-train vocabulary')
    report = read(BASE + 'report.json')
    verify_hashes({BASE + 'selection.json': report['selection_sha256']})
    require(report['team_counts'] == {k: len(v) for k, v in splits.items()}, 'Wrong split counts')
    rows = [json.loads(line) for line in (ROOT / BASE / 'test_predictions.jsonl').read_text().splitlines()]
    expected = {(i, hidden) for i, team in enumerate(splits['test']) for hidden in team}
    observed, scores = set(), {name: [] for name in report['metrics']}
    oov = unknown_visible = 0
    for row in rows:
        key = (row['team_id'], row['target'])
        require(key in expected and key not in observed, 'Missing/duplicated prediction mask')
        observed.add(key)
        team = splits['test'][row['team_id']]
        require(Counter(row['visible'] + [row['target']]) == Counter(team), 'Wrong visible members')
        covered = row['target'] in vocabulary
        require(row['target_in_vocabulary'] == covered, 'Incorrect OOV flag')
        oov += not covered
        unknown_visible += any(s not in vocabulary for s in row['visible'])
        require(set(row['methods']) == set(scores), 'Unexpected prediction methods')
        for method, prediction in row['methods'].items():
            names = [name for name, _ in prediction['top5']]
            probabilities = [p for _, p in prediction['top5']]
            require(len(names) == len(set(names)) == 5, 'Invalid top-5 length')
            require(set(names) <= vocabulary and not set(names) & set(row['visible']), 'Invalid top-5 species')
            require(all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities), 'Invalid probability')
            require(probabilities == sorted(probabilities, reverse=True) and sum(probabilities) <= 1 + 1e-12,
                    'Invalid top-5 probabilities')
            p = prediction['target_probability']
            require(np.isfinite(p) and 0 <= p <= 1, 'Invalid target probability')
            require(covered or p == 0, 'OOV target has vocabulary probability')
            if row['target'] in names:
                require(np.isclose(p, probabilities[names.index(row['target'])], atol=1e-12, rtol=0),
                        'Target probability differs from ranking')
            scores[method].append([row['target'] == names[0], row['target'] in names,
                                   -np.log(max(p, 1e-15)) if covered else np.nan])
    require(observed == expected and len(rows) == report['test_masks'], 'Incomplete masks')
    require(oov == report['out_of_vocabulary_targets'], 'Wrong OOV count')
    require(unknown_visible == report['masks_with_unknown_visible_species'], 'Wrong visible OOV count')
    metrics = ('top1', 'top5', 'log_loss_covered')
    for method, values in scores.items():
        reconstructed = np.nanmean(values, axis=0)
        expected_metrics = [report['metrics'][method][k] for k in metrics]
        require(np.allclose(reconstructed[:2], expected_metrics[:2], rtol=0, atol=1e-12),
                f'Top-k metrics differ from saved predictions: {method}')
        # Original neural log() ran in float32, JSON reconstruction runs in
        # float64. Keep a separate, explicit tolerance for this rounding only.
        require(abs(reconstructed[2] - expected_metrics[2]) < 1e-7,
                f'Log loss differs from saved predictions: {method}')

    catalog = read('data/processed/dataset_v1/catalog.json')
    prior = set.union(*groups.values()) | compositions([t['species_composition'] for t in catalog['teams'].values()])
    external_counts = {}
    for shard, protocol_path, result_path in (
        (31, 'config/team_profiles_external_v1.json', BASE + 'profiles/external_validation.json'),
        (32, 'config/rdf2vec_external_v1.json', BASE + 'rdf2vec/report.json'),
    ):
        protocol = read(protocol_path)
        verify_hashes(protocol.get('inputs_sha256', protocol.get('input_sha256', {})))
        test_path = DATA + f'external_metamon{shard}.json'
        test, result = read(test_path), read(result_path)
        verify_hashes({protocol_path: test['protocol_sha256'], test_path: result['test_sha256']})
        require(test['source_url'] == protocol['source_url'], 'Wrong source URL')
        if shard == 31:
            verify_hashes({protocol_path: result['protocol_sha256'],
                           'scripts/run_team_profiles_external.py': result['code_sha256']})
        else:
            require(result['selection'] == read(BASE + 'rdf2vec/selection.json'), 'RDF2Vec selection changed')
            verify_hashes({protocol_path: result['selection']['protocol_sha256']})
        current = compositions(test['teams'])
        require(not current & prior, 'External holdout overlaps previous compositions')
        prior |= current
        external_counts[str(shard)] = len(current)
        require(result['test_teams'] == len(current) and result['test_masks'] == 6 * len(current), 'Wrong test size')
        require(result['out_of_vocabulary_targets'] == sum(s not in vocabulary for t in current for s in t),
                'Wrong external OOV count')

    profile = read(BASE + 'profiles/audit.json')
    verify_hashes({'data/processed/dataset_v1/catalog.json': profile['catalog_sha256']})
    train_ids = {i for t in catalog['teams'].values() if t['split'] == 'train' for i in t['set_ids']}
    species_counts = Counter(r['base_species'] for r in catalog['sets'] if r['id'] in train_ids)
    require(profile['sets'] == len(train_ids) and profile['species'] == len(species_counts), 'Profile catalog differs')
    require(set(profile['profiles']) == set(species_counts), 'Profile species differ')
    for name, entry in profile['profiles'].items():
        require(entry['sets'] == species_counts[name], 'Profile has incorrect set count')
        require(len(entry['kb']) == 12 and all(0 <= p <= 1 for p in entry['kb']), 'Invalid KB fractions')
        for k in profile['candidates']:
            p = entry[f'k{k}']
            require(len(p) == k and all(0 <= x <= 1 for x in p) and abs(sum(p) - 1) < 1e-12,
                    'Invalid cluster fractions')
    verify_hashes(read(BASE + 'profiles/validation.json')['hashes'])
    scenarios = read(BASE + 'solver_scenarios.json')
    verify_hashes(scenarios['hashes'])
    require(len(scenarios['runs']) == scenarios['requests'], 'Missing solver requests')
    for method in ('top1', 'top5'):
        runs = [r[method] for r in scenarios['runs']]
        require(sum(r['legal'] for r in runs) == scenarios['methods'][method]['legal'], 'Wrong legal count')
        require(sum(r['legal'] and r['status'] == 'OPTIMAL' for r in runs)
                == scenarios['methods'][method]['optimal'], 'Wrong optimal count')
    return {'status': 'ok', 'split_teams': {k: len(v) for k, v in splits.items()},
            'prediction_masks_checked': len(rows), 'metrics_reconstructed': list(scores),
            'log_loss_reconstruction_absolute_tolerance': 1e-7,
            'external_compositions': external_counts, 'train_profile_sets': len(train_ids),
            'solver_requests': len(scenarios['runs']),
            'scope': 'Frozen hashes, unordered split separation, training vocabulary, all saved completion masks and metrics, external manifests, profile fractions and solver counts.',
            'limits': 'No refit, new battles, or replay revalidation. External KB/RDF2Vec reports do not save per-mask predictions, so their intervals are not independently reconstructed here.',
            'script_sha256': sha256(Path(__file__))}


if __name__ == '__main__':
    result = audit()
    save_json(ROOT / 'reports/extensions_audit.json', result)
    print(json.dumps(result, indent=2))
