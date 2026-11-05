import json
from pathlib import Path
from subprocess import TimeoutExpired
import pytest

from icon_pokemon.adapter import showdown
from icon_pokemon.validation import TeamValidator


def test_persistent_validator_matches_single_request_and_preserves_inputs():
    path = Path(__file__).resolve().parents[1] / 'data/processed/dataset_v1/catalog.json'
    catalog = json.loads(path.read_text())
    team = next(t for t in catalog['teams'].values() if t['split'] == 'train')['sets']
    original = json.dumps(team, sort_keys=True)
    with TeamValidator() as validator:
        assert validator.version == catalog['showdown_version']
        for candidate in (team, [team[0]] * 6, team):
            expected = showdown('validate-team.cjs', {'format': catalog['format'], 'sets': candidate})
            actual = validator.validate(catalog['format'], candidate, 5)
            assert actual == {k: expected[k] for k in ('legal', 'errors')}
        with pytest.raises(TimeoutExpired):
            validator.validate(catalog['format'], team, 0)
        assert validator.validate(catalog['format'], team, 5)['legal']
    assert json.dumps(team, sort_keys=True) == original
    assert validator.process is None
