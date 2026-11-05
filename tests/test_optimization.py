from itertools import combinations

import pytest

from icon_pokemon.optimization import solve_team
from icon_pokemon.matchups import coverage_score


@pytest.fixture
def catalog():
    records = [{"id": identifier, "base_species": species} for identifier, species in
               [("a", "A"), ("b", "B"), ("b2", "B"), ("c", "C"), ("d", "D")]]
    evidence = {"a": {}, "b": {}, "b2": {"hazard_removal": ["rapidspin"]},
                "c": {"hazard_setting": ["spikes"], "recovery": ["recover"]},
                "d": {"hazard_removal": ["defog"], "priority_attack": ["quickattack"]}}
    return records, evidence


def test_repair_matches_independent_exhaustive_objective(catalog):
    records, evidence = catalog
    original = {"a", "b", "c"}
    feasible = []
    for team in combinations(records, 3):
        ids = {r["id"] for r in team}
        coverage = set().union(*(evidence[i] for i in ids))
        if "a" not in ids or len({r["base_species"] for r in team}) != 3:
            continue
        if "hazard_removal" not in coverage:
            continue
        feasible.append(((len(original - ids), -len(coverage)), ids))
    best = min(score for score, _ in feasible)
    result = solve_team(records, evidence, size=3, original=sorted(original),
                        fixed=["a"], required=["hazard_removal"])
    selected = set(result.selected_ids)
    assert result.status == "OPTIMAL"
    assert selected in [ids for score, ids in feasible if score == best]
    assert result.replacements == best[0] == 1
    assert "a" in selected
    assert result.objective == result.best_bound


def test_species_clause_and_unavailable_requirement_are_infeasible(catalog):
    records, evidence = catalog
    result = solve_team(records, evidence, size=3, fixed=["b", "b2"])
    assert result.status == "INFEASIBLE"
    assert result.selected_ids == []
    assert result.objective is None
    without_removal = {i: {c: moves for c, moves in e.items() if c != "hazard_removal"}
                       for i, e in evidence.items()}
    assert solve_team(records, without_removal, size=3,
                      required=["hazard_removal"]).status == "INFEASIBLE"


@pytest.mark.parametrize("kwargs", [
    {"fixed": ["unknown"]}, {"original": ["unknown", "b", "c"]},
    {"required": ["strength"]}, {"original": ["a", "a", "c"]},
    {"original": ["a", "b"]},
])
def test_invalid_requests_are_rejected(catalog, kwargs):
    with pytest.raises(ValueError):
        solve_team(*catalog, size=3, **kwargs)


def test_duplicate_candidate_ids_are_rejected(catalog):
    records, evidence = catalog
    with pytest.raises(ValueError, match="Duplicate"):
        solve_team(records + [records[0]], evidence, size=3)


def test_matchup_repair_matches_exhaustive_lexicographic_oracle(catalog):
    records, evidence = catalog
    offense = {'a': [], 'b': ['t1'], 'b2': ['t1', 't2'], 'c': ['t2'], 'd': ['t3']}
    defense = {'a': ['t1'], 'b': ['t2'], 'b2': [], 'c': ['t3'], 'd': ['t1', 't2']}
    matrix = {r['id']: {t: {'offensive': t in offense[r['id']],
                            'defensive': t in defense[r['id']]}
                        for t in ['t1', 't2', 't3']} for r in records}
    original = {'a', 'b', 'c'}
    feasible = []
    for team in combinations(records, 3):
        ids = {r['id'] for r in team}
        if 'a' not in ids or len({r['base_species'] for r in team}) != 3:
            continue
        # Independent calculation, without reusing the implementation's score.
        counts = [sum(t in offense[i] for i in ids) for t in ['t1', 't2', 't3']]
        covered = sum(n > 0 for n in counts)
        if covered < 3:
            continue
        protected = sum(any(t in defense[i] for i in ids) for t in ['t1', 't2', 't3'])
        redundant = sum(min(n, 2) for n in counts)
        feasible.append(((len(original - ids), -covered, -protected, -redundant), ids))
    best = min(score for score, _ in feasible)
    result = solve_team(records, evidence, size=3, original=sorted(original), fixed=['a'],
                        matchups=matrix, minimum_offensive=3)
    assert result.status == 'OPTIMAL'
    assert set(result.selected_ids) in [ids for score, ids in feasible if score == best]
    assert result.replacements == best[0] == 2
    assert result.coverage_objective == tuple(-v for v in best[1:])
    assert result.coverage_objective == coverage_score(result.selected_ids, matrix)
    assert result.objective == result.best_bound
    # Without a quality requirement the unchanged original is a valid minimal repair.
    unchanged = solve_team(records, evidence, size=3, original=sorted(original), matchups=matrix)
    assert unchanged.replacements == 0
