"""Finite-catalog construction and lexicographic repair, not a win-rate estimator."""
from dataclasses import dataclass
from time import perf_counter
from ortools.sat.python import cp_model
from .knowledge import CAPABILITIES, ALL_CAPABILITIES
from .matchups import coverage_score


@dataclass
class TeamResult:
    status: str
    selected_ids: list[str]
    replacements: int | None
    covered_capabilities: list[str]
    wall_seconds: float
    objective: float | None
    best_bound: float | None
    coverage_objective: tuple[int, int, int] | None = None


def solve_team(records, evidence, *, required=(), fixed=(), original=(),
               size=6, seed=11, time_limit=10.0, matchups=None,
               minimum_offensive=0, minimum_defensive=0, excluded=(),
               maximum_replacements=None, on_solution=None, diagnostics=None):
    start = perf_counter()
    ids = {r["id"] for r in records}
    if len(ids) != len(records):
        raise ValueError("Duplicate candidate IDs")
    if not set(fixed).union(original) <= ids:
        raise ValueError("Fixed/original IDs must belong to the validated catalog")
    if not set(required) <= set(ALL_CAPABILITIES):
        raise ValueError("Unknown requested capability")
    if len(set(original)) != len(original) or (original and len(original) != size):
        raise ValueError("Original team must contain exactly size distinct sets")
    if size < 1 or time_limit <= 0:
        raise ValueError('Size and time limit must be positive')
    if matchups is None and (minimum_offensive or minimum_defensive):
        raise ValueError('Coverage thresholds require a matchup matrix')
    model = cp_model.CpModel()
    x = {r["id"]: model.new_bool_var(r["id"]) for r in records}
    model.add(sum(x.values()) == size)
    for species in sorted({r["base_species"] for r in records}):
        model.add(sum(x[r["id"]] for r in records if r["base_species"] == species) <= 1)
    for identifier in fixed:
        model.add(x[identifier] == 1)
    coverage = {}
    for capability in ALL_CAPABILITIES:
        coverage[capability] = model.new_bool_var(capability)
        candidates = [x[r["id"]] for r in records if capability in evidence[r["id"]]]
        if candidates:
            model.add_max_equality(coverage[capability], candidates)
        else:
            model.add(coverage[capability] == 0)
        if capability in required:
            model.add(coverage[capability] == 1)
    retained = sum(x[i] for i in original)
    for team in excluded:
        members = set(team)
        if len(members) != size or not members <= ids:
            raise ValueError('Excluded teams must have exactly size known members')
        model.add(sum(x[i] for i in members) <= size - 1)
    if maximum_replacements is not None:
        if not original or not 0 <= maximum_replacements <= size:
            raise ValueError('Replacement cap requires a valid original team and bound')
        model.add(retained >= size - maximum_replacements)
    if matchups is None:
        # Preserve the small capability demo; main experiments pass a matchup matrix.
        model.maximize((len(CAPABILITIES) + 1) * retained
                       + sum(coverage[c] for c in CAPABILITIES))
    else:
        if set(matchups) != ids:
            raise ValueError('Matchup candidates must exactly match the solver catalog')
        threats = sorted(next(iter(matchups.values()), {}))
        if not threats or any(set(row) != set(threats) for row in matchups.values()):
            raise ValueError('Matchup rows must have the same nonempty opponent axis')
        n = len(threats)
        if not 0 <= minimum_offensive <= n or not 0 <= minimum_defensive <= n:
            raise ValueError('Coverage thresholds outside the opponent axis')
        offense, defense, redundancy = [], [], []
        for threat in threats:
            for key, variables in [('offensive', offense), ('defensive', defense)]:
                covered = model.new_bool_var(f'{key}_{threat}')
                answers = [x[i] for i in x if matchups[i][threat][key]]
                if answers:
                    model.add_max_equality(covered, answers)
                else:
                    model.add(covered == 0)
                variables.append(covered)
            count = model.new_int_var(0, 2, f'redundancy_{threat}')
            model.add_min_equality(count, [2, sum(x[i] for i in x if matchups[i][threat]['offensive'])])
            redundancy.append(count)
        model.add(sum(offense) >= minimum_offensive)
        model.add(sum(defense) >= minimum_defensive)
        # Each integer weight exceeds the complete range of all lower objectives.
        defense_weight = 2 * n + 1
        offense_weight = n * defense_weight + 2 * n + 1
        retention_weight = n * offense_weight + n * defense_weight + 2 * n + 1
        model.maximize(retention_weight * retained + offense_weight * sum(offense)
                       + defense_weight * sum(defense) + sum(redundancy))
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = 1
    if diagnostics is not None:
        diagnostics.update(variables=len(model.proto.variables), constraints=len(model.proto.constraints),
                           model_seconds=perf_counter() - start)
    remaining = time_limit - (perf_counter() - start)
    if remaining <= 0:
        return TeamResult('UNKNOWN', [], None, [], perf_counter() - start, None, None)
    solver.parameters.max_time_in_seconds = remaining
    class Trace(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            chosen = sorted(i for i in x if self.value(x[i]))
            if on_solution(chosen, self.objective_value, self.best_objective_bound,
                           perf_counter() - start) is False:
                self.stop_search()

    status = solver.solve(model, Trace() if on_solution is not None else None)
    if diagnostics is not None:
        diagnostics.update(branches=solver.num_branches, conflicts=solver.num_conflicts,
                           solver_wall_seconds=solver.wall_time)
    success = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    selected = sorted(i for i in x if success and solver.value(x[i]))
    return TeamResult(
        solver.status_name(status), selected,
        len(set(original) - set(selected)) if success and original else None,
        sorted(c for c in coverage if success and solver.value(coverage[c])),
        perf_counter() - start,
        solver.objective_value if success else None,
        solver.best_objective_bound if success else None,
        coverage_score(selected, matchups) if success and matchups is not None else None,
    )
