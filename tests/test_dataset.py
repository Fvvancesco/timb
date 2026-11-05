from collections import Counter
import pytest
from icon_pokemon.dataset import partition_teams, make_jobs


def test_split_and_jobs_keep_replicates_and_team_groups_together():
    teams=[{'id':f't{i}', 'species_composition':[f'species{i}']} for i in range(12)]
    parts=partition_teams(teams,sizes=(4,4,4))
    jobs=make_jobs(parts,pair_counts=(3,3,3),roots=(11,23))
    owner={t['id']:split for split,ts in parts.items() for t in ts}
    assert len(jobs)==36 and len({j['job_id'] for j in jobs})==36
    pair_split={}
    for j in jobs:
        assert owner[j['team_a_id']]==owner[j['team_b_id']]==j['split']
        assert pair_split.setdefault(j['pair_id'],j['split'])==j['split']
    assert set(Counter(j['pair_id'] for j in jobs).values())=={4}
    assert set(owner)=={j[k] for j in jobs for k in ('team_a_id','team_b_id')}
    for j in jobs:
        reverse=next(k for k in jobs if k['pair_id']==j['pair_id'] and k['root_seed']==j['root_seed'] and k['orientation']!=j['orientation'])
        assert reverse['team_a_id']==j['team_b_id']
        assert reverse['engine_seed']==j['engine_seed']
    assert make_jobs(parts,pair_counts=(3,3,3),roots=(11,23))==jobs


def test_reject_composition_duplicates_across_splits():
    with pytest.raises(ValueError,match='composition'):
        partition_teams([{'id':'a','species_composition':['x']},
                         {'id':'b','species_composition':['x']}],sizes=(2,0,0))
