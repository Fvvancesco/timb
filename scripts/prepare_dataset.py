"""Freeze a small legal reconstructed-team corpus and splits before any labels."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icon_pokemon.adapter import showdown, sha256, save_json
from icon_pokemon.dataset import stable_id, partition_teams, make_jobs


def read_sources(archive, names=None):
    with tarfile.open(archive, 'r:gz') as tar:
        members = {m.name:m for m in tar.getmembers() if m.isfile() and m.name.endswith('.gen9ou_team')}
        return [{'path':name, 'text':tar.extractfile(members[name]).read().decode()}
                for name in (names if names is not None else sorted(members))]


def main():
    output = ROOT / 'data/processed/dataset_v1'
    manifest_path = ROOT / 'config/dataset_v1_manifest.json'
    if manifest_path.exists():
        raise SystemExit('Manifest already frozen. Use a new dataset version instead of overwriting splits.')
    development_archive = ROOT / 'data/raw/teams/competitive-gen9ou.tar.gz'
    development = showdown('export-catalog.cjs', {'format':'gen9ou', 'teams':read_sources(development_archive)})
    excluded_groups = {tuple(t['species_composition']) for t in development['teams']}
    archive = ROOT / 'data/raw/teams/modern_replays-gen9ou.tar.gz'
    with tarfile.open(archive, 'r:gz') as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile() and m.name.endswith('.gen9ou_team'))
    random.Random(20261103).shuffle(names)
    selected = []; records = {}; groups = set(); decisions = []; inspected = 0
    # Cap inspection rather than scanning the whole corpus without a useful stopping point.
    for offset in range(0, min(len(names), 2000), 100):
        result = showdown('export-catalog.cjs', {'format':'gen9ou',
            'teams':read_sources(archive, names[offset:offset + 100])}, timeout=120)
        batch_records = {r['id']:r for r in result['sets']}
        for team in result['teams']:
            inspected += 1
            group = tuple(team['species_composition'])
            reason = None
            if not team['legal']: reason = 'illegal_under_pinned_rules'
            elif len(team['set_ids']) != 6 or any(len(batch_records[i]['moves']) != 4 for i in team['set_ids']): reason = 'incomplete_six_by_four'
            elif group in excluded_groups: reason = 'development_species_composition'
            elif group in groups: reason = 'duplicate_species_composition'
            if reason:
                decisions.append({'source':team['source'], 'accepted':False,
                                  'reason':reason, 'validation_errors':team['validation_errors']})
                continue
            identifier = stable_id(sorted(team['set_ids']))
            selected.append({'id':identifier, 'source':team['source'],
                'species_composition':list(group), 'set_ids':team['set_ids']})
            groups.add(group)
            for i in team['set_ids']: records.setdefault(i, batch_records[i])
            decisions.append({'source':team['source'], 'accepted':True, 'team_id':identifier})
            if len(selected) == 140: break
        if len(selected) == 140: break
    if len(selected) != 140:
        raise SystemExit(f'Only {len(selected)} eligible team groups in bounded scan; no manifest written')
    partitions = partition_teams(selected)
    jobs = make_jobs(partitions)
    teams = {}
    for split, values in partitions.items():
        for team in values:
            payload = [records[i]['set'] for i in team['set_ids']]
            validated = showdown('validate-team.cjs', {'format':'gen9ou', 'sets':payload})
            if not validated['legal']: raise RuntimeError(validated['errors'])
            teams[team['id']] = {**team, 'split':split, 'sets':payload,
                'packed':validated['packed'], 'text':validated['text'].strip()}
    catalog = {'format':'gen9ou','showdown_version':result['showdown_version'],
               'sets':sorted(records.values(),key=lambda r:r['id']), 'teams':teams}
    save_json(output / 'catalog.json', catalog)
    save_json(output / 'selection_audit.json', {'inspected':inspected,'decisions':decisions})
    manifest = {'dataset_version':'v1', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'format':'gen9ou', 'showdown_version':result['showdown_version'], 'pilot':'foul-play',
        'pilot_commit':'8f541f80f91f960603673796559f796b10ccf0fb',
        'search_time_ms':25,'search_parallelism':1,'search_threads':1,
        'max_turns':200, 'wall_limit_seconds':120,
        'planned_max_jobs':1000, 'initial_authorized_train_jobs':20,
        'label':'Individual simulated battle outcome conditional on declared policies and rules; not exact pair win rate',
        'source_kind':'Metamon reconstructed replay teams; observed full human sets are NOT asserted',
        'sampling':'One legal full set-team per species-composition group, deterministic shuffled source order; excludes all competitive development compositions; not usage-weighted',
        'split_sizes':{k:len(v) for k,v in partitions.items()},
        'job_counts':dict(Counter(j['split'] for j in jobs)),
        'seed_policy':'Actual engine seed must be injected and verified; five roots and both sides per unordered pair. Foul Play timed/native search may remain nondeterministic; archive inputLog for trajectory replay.',
        'censoring':'Timeout/error/forced stop are null outcomes; natural draws separate; never silently discard incomplete jobs',
        'catalog_path':'data/processed/dataset_v1/catalog.json',
        'hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [archive,development_archive,
            output / 'catalog.json', ROOT/'tools/showdown/package-lock.json',
            ROOT/'tools/showdown/export-catalog.cjs',ROOT/'scripts/prepare_dataset.py',
            ROOT/'src/icon_pokemon/dataset.py']},
        'teams':{i:{k:v for k,v in t.items() if k not in ('sets','packed','text')} for i,t in teams.items()},
        'jobs':jobs}
    save_json(manifest_path,manifest)
    print(json.dumps({'selected_teams':len(teams),'inspected':inspected,
                      'splits':manifest['split_sizes'],'jobs':manifest['job_counts']},indent=2))


if __name__ == '__main__': main()
