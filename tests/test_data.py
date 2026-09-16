"""Offline unit tests. Synthetic schedules exist only as validator test inputs."""
from pathlib import Path
import pandas as pd
import pytest
from champions import data


def test_public_contract():
    assert callable(getattr(data, 'fetch_current', None))
    assert callable(getattr(data, 'fetch_history', None))
    assert callable(getattr(data, 'validate_fixtures', None))


@pytest.fixture
def fixtures():
    teams = sorted(data.CURRENT_TEAMS)
    rows = []
    for day in range(8):
        for i in range(18):
            home, away = teams[i], teams[18 + (i + day) % 18]
            if day % 2:
                home, away = away, home
            rows.append(dict(match_id=f'test-{day}-{i}', date=f'2026-10-{day+1:02d}T20:00:00Z',
                season='2026/27', stage='league', matchday=day+1, home=home, away=away,
                home_goals=1 if day == 0 else None, away_goals=0 if day == 0 else None,
                status='finished' if day == 0 else 'scheduled', source='synthetic unit test only'))
    return pd.DataFrame(rows)


def test_balanced_schedule(fixtures):
    data.validate_fixtures(fixtures)
    assert len(fixtures) == 144
    assert (fixtures.status == 'finished').sum() == 18
    assert fixtures.iloc[0].home == 'AEK Athens'


def test_uefa_parser_real_excerpt():
    text = '''Matchday 1
Tuesday 8 September 2026
AEK Athens 1-0 LASK
Highlights: Club Brugge 2-3 Aston Villa
Matchday 2
Tuesday 13 October 2026
Lens vs Sporting CP (18:45)
'''
    frame = data._parse_current(text)
    assert len(frame) == 2
    assert frame.iloc[0]['date'] == '2026-09-08T20:00:00Z'
    assert frame.iloc[1]['date'] == '2026-10-13T17:45:00Z'
    assert frame.iloc[1]['status'] == 'scheduled'


def test_optional_real_source_integration():
    path = Path('data/raw/uefa-2026-27.html')
    if not path.exists():
        pytest.skip('Optional downloaded UEFA integration cache')
    data.validate_fixtures(data._parse_current(path.read_text(encoding='utf-8')))


@pytest.mark.parametrize('column,value', [
    ('home', None), ('date', '2026-02-30T20:00:00Z'),
    ('date', '2026-09-08'), ('home_goals', -1), ('home_goals', 1.5),
    ('home_goals', float('inf')), ('home_goals', None),
    ('matchday', 1.5), ('status', 'unknown'), ('source', ''),
    ('season', '2025/26'), ('stage', 'final'),
])
def test_reject_bad_fields(fixtures, column, value):
    fixtures[column] = fixtures[column].astype(object)
    fixtures.loc[0, column] = value
    with pytest.raises(ValueError):
        data.validate_fixtures(fixtures)


def test_reject_missing_duplicate_and_unbalanced(fixtures):
    with pytest.raises(ValueError):
        data.validate_fixtures(fixtures.iloc[:-1])
    bad = fixtures.copy()
    bad.iloc[-1] = bad.iloc[0]
    with pytest.raises(ValueError):
        data.validate_fixtures(bad)
    bad = fixtures.copy()
    bad.loc[0, ['home', 'away']] = bad.loc[0, ['away', 'home']].to_numpy()
    with pytest.raises(ValueError):
        data.validate_fixtures(bad)


def test_scheduled_scores_rejected(fixtures):
    fixtures.loc[fixtures.status == 'scheduled', 'home_goals'] = 0
    with pytest.raises(ValueError):
        data.validate_fixtures(fixtures)


def test_aliases():
    for alias, canonical in [('Man United', 'Manchester United'),
                             ('Bayern Munich', 'Bayern München'),
                             ('FC Barcelona', 'Barcelona'),
                             ('Sabah FK', 'Sabah'), ('Bodo/Glimt', 'Bodø/Glimt')]:
        assert data.normalize_team(alias) == canonical


def test_openfootball_regulation_score_and_year_rollover():
    text = '''= UEFA Champions League 2025/26
  Tue Dec 9 2025
    21:00 FC Barcelona (ESP) v Arsenal FC (ENG) 2-1 (1-0)
  Wed Jan 21 2026
    21:00 Arsenal FC (ENG) v FC Barcelona (ESP) 3-2 a.e.t. (2-2, 1-1)
'''
    result = data._parse_openfootball(text, 'test-source', 'UCL', 2025)
    assert len(result) == 2
    assert result[1]['home_goals'] == 2
    assert result[1]['away_goals'] == 2
    assert result[1]['date'].startswith('2026-01-21')


def test_history_rejects_invalid_results():
    frame = pd.DataFrame([dict(date='2025-01-01T00:00:00Z', home='Arsenal',
        away='Barcelona', home_goals=1.5, away_goals=0, competition='UCL', source='test')])
    with pytest.raises(ValueError):
        data.validate_history(frame)
