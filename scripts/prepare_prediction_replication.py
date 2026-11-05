"""Freeze a separate evaluation of the existing predictor on disjoint new teams."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import json
from random import Random
import sys
import tarfile

import joblib
import numpy as np

from run_coverage_confirmation import ROOT, dump

sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import sha256, showdown
from icon_pokemon.dataset import engine_seed, stable_id
from icon_pokemon.features import TeamFeatures
from icon_pokemon.knowledge import build_graph, capability_evidence
from icon_pokemon.supervised import probabilities

OUTPUT = ROOT / 'data/processed/prediction_replication_v2'
SHARDS = {'workstation': (112, 14), 'laptop': (53, 6), 'casa': (34, 4),
          'bazzite': (48, 6), 'windows': (53, 6)}
PAIR_COUNT = sum(count for count, _ in SHARDS.values())
SAMPLE_SEED = 251108
ROOT_SEED = 883


def main():
    manifest_path = OUTPUT / 'manifest.json'
    if manifest_path.exists():
        raise ValueError('Replication already frozen')
    data = ROOT / 'data/processed'
    catalog_path = data / 'dataset_v1/catalog.json'
    archive_path = ROOT / 'data/raw/teams/modern_replays-gen9ou.tar.gz'
    prior_paths = [data / f'coverage_ablation_v{i}/manifest.json' for i in (1, 2)]
    original_path = data / 'final_v1/manifest.json'
    original = json.loads(original_path.read_text())
    model_path = ROOT / original['model_path']
    if sha256(model_path) != original['model_sha256']:
        raise ValueError('Frozen model hash differs')
    bundle = joblib.load(model_path)
    if (bundle['scikit_learn'] != version('scikit-learn')
            or sha256(catalog_path) != bundle['input_hashes'][str(catalog_path.relative_to(ROOT))]
            or any(sha256(ROOT / p) != digest for p, digest in bundle['code_hashes'].items())):
        raise ValueError('Frozen feature inputs or software differ')

    catalog = json.loads(catalog_path.read_text())
    original_full_path = data / 'final_v1/catalog.json'
    if sha256(original_full_path) != original['hashes'][str(original_full_path.relative_to(ROOT))]:
        raise ValueError('Original final catalog hash differs')
    original_full = json.loads(original_full_path.read_text())
    excluded = {tuple(sorted(t['species_composition'])) for t in original_full['teams'].values()}
    prior_sources = set()
    for path in prior_paths:
        for request in json.loads(path.read_text())['requests']:
            prior_sources.add(request['opponent'])
            if 'opponent_composition' in request:
                excluded.add(tuple(request['opponent_composition']))
    prior_replication_path = data / 'prediction_replication_v1/manifest.json'
    prior_replication = json.loads(prior_replication_path.read_text())
    for pair in prior_replication['requests']:
        excluded.update(tuple(comp) for comp in pair['compositions'])
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive if m.isfile() and m.name.endswith('.gen9ou_team')]
        ordered = sorted(members, key=lambda member: member.name)
        sample = Random(SAMPLE_SEED).sample(ordered, min(4000, len(ordered)))
        previous = [m for m in members if m.name in prior_sources]
        if len(previous) != len(prior_sources):
            raise ValueError('Previous team source missing')
        selected = {m.name for m in sample} | prior_sources
        sources = {m.name: archive.extractfile(m).read().decode() for m in members if m.name in selected}

    def export(names):
        return showdown('export-catalog.cjs', {'format': catalog['format'],
            'teams': [{'path': name, 'text': sources[name]} for name in names]}, timeout=120)

    for i in range(0, len(previous), 100):
        for team in export([m.name for m in previous[i:i + 100]])['teams']:
            excluded.add(tuple(team['species_composition']))
    candidates = [m.name for m in sample if m.name not in prior_sources]
    teams, fresh_records = [], {}
    for i in range(0, len(candidates), 100):
        result = export(candidates[i:i + 100])
        sets = {record['id']: record for record in result['sets']}
        for team in result['teams']:
            composition = tuple(team['species_composition'])
            ids = team['set_ids']
            if (not team['legal'] or len(ids) != 6 or composition in excluded
                    or any(len(sets[identifier]['moves']) != 4 for identifier in ids)):
                continue
            teams.append(team)
            excluded.add(composition)
            fresh_records.update({identifier: sets[identifier] for identifier in ids})
            if len(teams) == 2 * PAIR_COUNT:
                break
        if len(teams) == 2 * PAIR_COUNT:
            break
    if len(teams) != 2 * PAIR_COUNT:
        raise ValueError(f'Only {len(teams)} new eligible compositions')
    Random(SAMPLE_SEED + 1).shuffle(teams)
    print(f'Selected {len(teams)} disjoint compositions', flush=True)

    def validate(team):
        sets = [fresh_records[i]['set'] for i in team['set_ids']]
        validated = showdown('validate-team.cjs', {'format': catalog['format'], 'sets': sets})
        if not validated['legal']:
            raise ValueError(f'Normalized team rejected: {team["source"]}')
        return {**team, 'sets': sets, 'packed': validated['packed'], 'text': validated['text']}

    with ThreadPoolExecutor(max_workers=6) as pool:
        teams = list(pool.map(validate, teams))

    records = {r['id']: r for r in catalog['sets']}
    records.update(fresh_records)
    payload = {'format': catalog['format'], 'move_ids':
               sorted({m['id'] for r in records.values() for m in r['moves']})}
    mechanics = showdown('export-mechanics.cjs', payload)
    semantics = showdown('export-move-semantics.cjs', payload)
    expanded = {**catalog, 'sets': list(records.values())}
    _, graph = build_graph(expanded, mechanics, move_semantics=semantics)
    evidence = {i: capability_evidence(graph, r) for i, r in records.items()}
    features = TeamFeatures(records, graph, mechanics, evidence, capabilities=bundle['capabilities'])
    if tuple(features.names) != tuple(bundle['features']):
        raise ValueError('Frozen feature schema differs')

    original_predictions = data / 'final_v1/predictions.json'
    if sha256(original_predictions) != original['hashes'][str(original_predictions.relative_to(ROOT))]:
        raise ValueError('Frozen predictions hash differs')
    old_pairs = [job for job in original['jobs'] if job['split'] == 'test']
    old_expected = {row['job_id']: row['p_team_a']
                    for row in json.loads(original_predictions.read_text())}
    old_x = [features.pair(*(catalog['teams'][job[k]]['set_ids'] for k in ('team_a_id', 'team_b_id')))
             for job in old_pairs]
    old_actual = probabilities(bundle['estimator'], old_x)
    if not np.allclose(old_actual, [old_expected[job['job_id']] for job in old_pairs], atol=1e-12, rtol=0):
        raise ValueError('The feature pipeline no longer reproduces frozen predictions')

    requests, jobs, inputs = [], [], []
    for index, (a, b) in enumerate(zip(teams[::2], teams[1::2], strict=True)):
        pair_id = f'pred-rep-v2-{index:03}'
        requests.append({'id': pair_id, 'compositions': [t['species_composition'] for t in (a, b)],
                         'sources': [t['source'] for t in (a, b)]})
        for side, (left, right) in enumerate(((a, b), (b, a))):
            job_id = f'{pair_id}-{ROOT_SEED}-{side}'
            token = stable_id([job_id])[:10]
            inputs.append(features.pair(left['set_ids'], right['set_ids']))
            jobs.append({'job_id': job_id, 'request_id': pair_id, 'root': ROOT_SEED,
                         'orientation': side, 'players': ['IR' + token + s for s in 'ab'],
                         'engine_seed': engine_seed(pair_id, ROOT_SEED),
                         'python_seeds': [int(stable_id([job_id, str(i)])[:8], 16) for i in (0, 1)],
                         'teams': [{**{k: t[k] for k in ('sets', 'packed')},
                                    'text': t['text'].strip() + '\n'} for t in (left, right)]})
    p = probabilities(bundle['estimator'], inputs)
    if not np.allclose(p[::2] + p[1::2], 1, atol=1e-12, rtol=0):
        raise ValueError('Predictions for mirrored orientations are not complementary')
    for job, prediction in zip(jobs, p, strict=True):
        job['p_team_a'] = float(prediction)

    runner = Path(__file__).with_name('run_coverage_confirmation.py')
    analysis = Path(__file__).with_name('analyze_prediction_replication_v2.py')
    paths = [catalog_path, archive_path, model_path,
             original_predictions, runner, analysis, Path(__file__)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dump(manifest_path, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'format': catalog['format'], 'source_sha256': {p.name: sha256(p) for p in paths},
        'prior_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                         [*prior_paths, original_path, original_full_path, prior_replication_path]},
        'pilot_protocol': {k: original[k] for k in ('pilot', 'pilot_commit', 'showdown_version',
            'search_time_ms', 'search_parallelism', 'search_threads', 'max_turns', 'wall_limit_seconds')},
        'protocol': 'Third, separate evaluation planned after the first two tests. Fixed cohort of 600 new teams in 300 disjoint pairs. No composition overlaps earlier prediction cohorts, the original catalog, final selector candidates or earlier coverage opponents. One root and two orientations per pair. Use all decisive, verified outcomes. Primary log loss difference from 0.5; secondary Brier, AUROC and confusion matrix at 0.5. 10,000 percentile bootstrap draws by pair, seed 11. No pooling, optional stopping or model tuning.',
        'sample_seed': SAMPLE_SEED, 'requests': requests, 'jobs': jobs})
    digest = sha256(manifest_path)
    (OUTPUT / 'assignments').mkdir(exist_ok=True)
    start = 0
    for shard, (count, workers) in SHARDS.items():
        dump(OUTPUT / 'assignments' / f'{shard}.json',
             {'shard': shard, 'workers': workers, 'manifest_sha256': digest,
              'requests': [r['id'] for r in requests[start:start + count]]})
        start += count
    print(f'Frozen {len(requests)} pairs and {len(jobs)} jobs at {digest}', flush=True)


if __name__ == '__main__':
    main()
