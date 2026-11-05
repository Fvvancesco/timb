"""Pre-battle team features shared by training and candidate selection."""
from collections import Counter
from functools import lru_cache
from math import prod
from statistics import mean

from .knowledge import CAPABILITIES, ALL_CAPABILITIES
from .matchups import build_matchups, coverage_score

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
TYPE_FIELDS = ('member', 'weak', 'resists', 'immune', 'attack')


def effective_stats(record, natures):
    spread = record['set']
    level = spread.get('level', 100)
    nature = natures[spread.get('nature') or 'Serious']
    stats = {}
    for stat in STATS:
        base = record['base_stats'][stat]
        iv = (spread.get('ivs') or {}).get(stat, 31)
        ev = (spread.get('evs') or {}).get(stat, 0)
        value = (2 * base + iv + ev // 4) * level // 100
        if stat == 'hp':
            stats[stat] = 1 if base == 1 else value + level + 10
        else:
            factor = 110 if nature['plus'] == stat else 90 if nature['minus'] == stat else 100
            stats[stat] = (value + 5) * factor // 100
    return stats


class TeamFeatures:
    def __init__(self, records, graph, mechanics, evidence, *, capabilities=None):
        self.records = records
        self.graph = graph
        self.mechanics = mechanics
        self.evidence = evidence
        self.capabilities = tuple(CAPABILITIES if capabilities is None else capabilities)
        if len(set(self.capabilities)) != len(self.capabilities) or not set(self.capabilities) <= ALL_CAPABILITIES.keys():
            raise ValueError('Feature capabilities must be distinct known capabilities')
        self.team_names = tuple(
            [f'{field}_{kind}' for field in TYPE_FIELDS for kind in mechanics['types']]
            + [f'{stat}_{summary}' for stat in STATS for summary in ('mean', 'min', 'max')]
            + list(self.capabilities)
        )
        self.direction_names = ('offense', 'defense', 'redundancy', 'worst_best',
                                'mean_best', 'threats_with_known_multiplier')
        self.names = tuple(f'{side}_{name}' for side in ('a', 'b')
                           for name in self.team_names + self.direction_names)

    @lru_cache(maxsize=2048)
    def team(self, ids):
        counts = {field: Counter() for field in TYPE_FIELDS}
        stats = []
        for identifier in ids:
            record = self.records[identifier]
            stats.append(effective_stats(record, self.mechanics['natures']))
            counts['member'].update(record['types'])
            attacks = self.mechanics['moves']
            counts['attack'].update({attacks[m['id']]['type'] for m in record['moves']
                                     if attacks[m['id']]['eligible']})
            for attack, interactions in self.mechanics['chart'].items():
                multiplier = prod(interactions[defense] for defense in record['types'])
                if multiplier == 0:
                    counts['immune'][attack] += 1
                elif multiplier < 1:
                    counts['resists'][attack] += 1
                elif multiplier > 1:
                    counts['weak'][attack] += 1
        values = [counts[field][kind] for field in TYPE_FIELDS for kind in self.mechanics['types']]
        for stat in STATS:
            members = [s[stat] for s in stats]
            values.extend((mean(members), min(members), max(members)))
        values.extend(sum(cap in self.evidence[i] for i in ids) for cap in self.capabilities)
        return tuple(values)

    def direction(self, ids, opponents, matrix):
        best = []
        for threat in opponents:
            values = [matrix[i][threat]['max_outgoing_multiplier'] for i in ids
                      if matrix[i][threat]['max_outgoing_multiplier'] is not None]
            if values:
                best.append(max(values))
        return (*coverage_score(ids, matrix), min(best, default=0),
                mean(best) if best else 0, len(best))

    def pair(self, team_a, team_b):
        a, b = tuple(sorted(team_a)), tuple(sorted(team_b))
        if not a or not b:
            raise ValueError('Both sides need at least one member')
        matrix = build_matchups(a, b, self.graph, self.mechanics)
        # The reverse offensive multiplier is already present in the forward matrix.
        reverse = {}
        for j in b:
            reverse[j] = {}
            for i in a:
                cell = matrix[i][j]
                incoming = cell['max_incoming_multiplier']
                outgoing = cell['max_outgoing_multiplier']
                reverse[j][i] = {
                    'offensive': incoming is not None and incoming > 1,
                    'defensive': outgoing is not None and outgoing <= 1,
                    'max_outgoing_multiplier': incoming,
                }
        return (*self.team(a), *self.direction(a, b, matrix),
                *self.team(b), *self.direction(b, a, reverse))
