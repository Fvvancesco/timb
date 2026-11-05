"""Confirm the frozen KB feature comparison on an unseen replay shard."""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from run_team_profile_extension import (
    CATALOG, DATA, MODEL_SEEDS, OUTPUT, REPORT, ROOT, fit_variant, input_matrix,
    masked_cases, metrics, profile_matrix,
)
from run_team_completion_extension import freeze
from icon_pokemon.adapter import save_json, sha256
from icon_pokemon.evaluation import grouped_interval

PROTOCOL = ROOT / 'config/team_profiles_external_v1.json'
TEST = ROOT / 'data/processed/team_completion_v1/external_metamon31.json'
RESULT = REPORT / 'external_validation.json'


def check_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in protocol['inputs_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    return protocol


def prior_teams():
    previous = json.loads(DATA.read_text())['splits']
    catalog = json.loads(CATALOG.read_text())['teams']
    used = {tuple(sorted(t)) for split in previous.values() for t in split}
    used.update(tuple(sorted(t['species_composition'])) for t in catalog.values())
    return previous['train'], used


def prepare(path, protocol):
    from run_human_expanded import ingest
    matches, species, counts = ingest([path], protocol['minimum_rating'])
    teams = {tuple(sorted(species[name]['base_species'] for name in side))
             for match in matches.values() for side in match['sides']}
    valid = {t for t in teams if len(t) == len(set(t)) == 6}
    _, used = prior_teams()
    test = sorted(valid - used)
    if len(test) < 2:
        raise ValueError('Insufficient independent compositions')
    freeze(TEST, {'source_url': protocol['source_url'], 'source_sha256': sha256(path),
                  'protocol_sha256': sha256(PROTOCOL), 'source_counts': counts,
                  'excluded_invalid_compositions': len(teams - valid),
                  'excluded_previous_compositions': len(valid & used), 'teams': [list(t) for t in test]})
    print(f'Test frozen: {len(test)} teams. Removed {len(valid & used)} previously used teams.', flush=True)


def evaluate(protocol):
    train_teams, used = prior_teams()
    test = json.loads(TEST.read_text())
    test_teams = [tuple(t) for t in test['teams']]
    if (test['protocol_sha256'] != sha256(PROTOCOL) or used.intersection(test_teams)
            or len(test_teams) != len(set(test_teams))):
        raise ValueError('Invalid external test manifest')
    selection = json.loads((OUTPUT / 'selection.json').read_text())
    vocab = selection['candidate_species']
    profiles = json.loads((REPORT / 'audit.json').read_text())['profiles']
    params = protocol['network']
    if tuple(params['seeds']) != MODEL_SEEDS:
        raise ValueError('Unexpected model seeds')
    train, train_y = masked_cases(train_teams, vocab)
    visible, target = masked_cases(test_teams, vocab)
    matrices = {
        'identity': (input_matrix(train, len(vocab)), input_matrix(visible, len(vocab))),
        'kb': (profile_matrix(train, vocab, profiles, 'kb'), profile_matrix(visible, vocab, profiles, 'kb')),
    }
    runs = dict(Parallel(n_jobs=2)(delayed(fit_variant)(
        name, x, train_y, tx, target, visible, len(vocab), params['dimension'], params['epochs'])
        for name, (x, tx) in matrices.items()))
    groups = np.repeat(np.arange(len(test_teams)), 6)
    covered = target >= 0
    delta = runs['kb'] - runs['identity']
    report = {'scope': 'New shard and previously unused compositions. Same public replay source family.',
              'source_url': protocol['source_url'], 'train_teams': len(train_teams),
              'test_teams': len(test_teams), 'test_masks': len(target),
              'out_of_vocabulary_targets': int((~covered).sum()),
              'profile_member_coverage': float(np.mean([name in profiles for team in test_teams for name in team])),
              'metrics': {k: metrics(v) for k, v in runs.items()},
              'paired_kb_minus_identity_top1_top5': grouped_interval(delta[:,:2], groups.tolist()),
              'paired_kb_minus_identity_log_loss': grouped_interval(delta[covered,2], groups[covered].tolist()),
              'protocol_sha256': sha256(PROTOCOL), 'test_sha256': sha256(TEST),
              'code_sha256': sha256(Path(__file__))}
    save_json(RESULT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','evaluate'))
    parser.add_argument('parquet', nargs='?', type=Path)
    args = parser.parse_args()
    protocol = check_protocol()
    if args.action == 'prepare':
        if args.parquet is None:
            parser.error('prepare requires the frozen source parquet')
        prepare(args.parquet, protocol)
    else:
        evaluate(protocol)
