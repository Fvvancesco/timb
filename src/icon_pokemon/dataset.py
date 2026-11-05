"""Deterministic team-group splits and battle jobs; no battle outcomes are read."""
import hashlib
import itertools
import random


def stable_id(parts):
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def engine_seed(pair_id, root_seed):
    value = hashlib.sha256(f"icon-dataset-v1|{pair_id}|{root_seed}|engine".encode()).digest()
    return [int.from_bytes(value[i:i + 2], "big") for i in range(0, 8, 2)]


def partition_teams(teams, sizes=(100, 20, 20), seed=20261103):
    if len(teams) != sum(sizes):
        raise ValueError("Team count must match the partition sizes")
    if len({t['id'] for t in teams}) != len(teams):
        raise ValueError("Duplicate teams")
    groups = [tuple(t['species_composition']) for t in teams]
    if len(set(groups)) != len(groups):
        raise ValueError("Species-composition groups must not cross partitions")
    shuffled = sorted(teams, key=lambda t: t['id'])
    random.Random(seed).shuffle(shuffled)
    result = {}
    offset = 0
    for name, size in zip(('train', 'validation', 'test'), sizes):
        result[name] = shuffled[offset:offset + size]
        offset += size
    return result


def sample_pairs(teams, count, seed):
    identifiers = sorted(t['id'] for t in teams)
    rng = random.Random(seed)
    rng.shuffle(identifiers)
    if len(identifiers) % 2 or not len(identifiers) // 2 <= count <= len(identifiers) * (len(identifiers) - 1) // 2:
        raise ValueError("Need an even number of teams and enough pairs to cover each team")
    # A random matching guarantees every reserved team is represented at least once.
    pairs = {tuple(sorted(identifiers[i:i + 2])) for i in range(0, len(identifiers), 2)}
    remaining = sorted(set(itertools.combinations(sorted(identifiers), 2)) - pairs)
    pairs.update(rng.sample(remaining, count - len(pairs)))
    return sorted(pairs)


def make_jobs(partitions, pair_counts=(70, 15, 15), roots=(11, 23, 37, 53, 71), *,
              explicit_pairs=None):
    jobs = []
    for index, (split, teams) in enumerate(partitions.items()):
        pairs = (sample_pairs(teams, pair_counts[index], 20261103 + index)
                 if explicit_pairs is None else explicit_pairs[split])
        for root in roots:
            for orientation in (0, 1):
                for a, b in pairs:
                    pair_id = stable_id([a, b])
                    side_ids = [a, b] if orientation == 0 else [b, a]
                    job_id = f"{split}-{pair_id}-{root}-{orientation}"
                    jobs.append({"job_id":job_id, "split":split, "pair_id":pair_id,
                        "root_seed":root, "orientation":orientation,
                        "team_a_id":side_ids[0], "team_b_id":side_ids[1],
                        "showdown_p1_team_id":side_ids[0], "showdown_p2_team_id":side_ids[1],
                        "engine_seed":engine_seed(pair_id, root),
                        "pilot_python_seeds":{side:int(stable_id([job_id, side])[:8],16)
                                              for side in ('p1','p2')}})
    return jobs
