from scripts.run_human_preview import read_preview


def test_human_preview_uses_pre_battle_rosters_and_pre_battle_ratings():
    log = ['|player|p1|Alpha|1|1620', '|player|p2|Beta|1|1690']
    for side in ('p1', 'p2'):
        log.extend(f'|poke|{side}|Species{i}, L100|' for i in range(6))
    log.extend(['|teampreview', '|start', '|poke|p1|HiddenForm|',
                '|player|p1|Alpha|1|2000', '|win|ALPHA'])
    row = {'id': 'gen9ou-12345', 'formatid': 'gen9ou', 'log': '\n'.join(log)}
    status, result = read_preview(row, 1600)
    assert status == 'parsed'
    assert result['id'] == '12345'
    assert result['team_a'] == [f'Species{i}' for i in range(6)]
    assert result['ratings'] == [1620, 1690]
    assert result['y'] == 1
    assert read_preview(row, 1700) == ('rating', None)
