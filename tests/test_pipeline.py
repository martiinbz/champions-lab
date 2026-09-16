import pandas as pd
import pytest

from champions.pipeline import as_of, training_data, sha256, write_json


def fixture():
    return pd.DataFrame([
        {"date": "2026-09-08T19:00:00Z", "matchday": 1, "home": "A", "away": "B", "home_goals": 1, "away_goals": 0, "status": "finished"},
        {"date": "2026-10-13T19:00:00Z", "matchday": 2, "home": "B", "away": "A", "home_goals": 0, "away_goals": 3, "status": "finished"},
    ])


def test_asof_does_not_leak_final_score_at_kickoff_or_future_matchday():
    frame = fixture()
    before = as_of(frame, "2026-09-08T20:00:00Z")
    assert before.home_goals.isna().all()
    after = as_of(frame, "2026-09-09T00:00:00Z")
    assert after.home_goals.tolist()[0] == 1
    assert pd.isna(after.home_goals.tolist()[1])
    assert frame.home_goals.notna().all()  # input remains immutable


def test_training_merge_deduplicates_current_match_and_excludes_future():
    frame = fixture()
    history = frame.copy()
    result = training_data(history, as_of(frame, "2026-09-16"), "2026-09-16")
    assert len(result) == 1
    assert result.home_goals.iloc[0] == 1
    assert (result.completed_at > result.date).all()


def test_conflicting_sources_are_rejected():
    frame = fixture()
    history = frame.copy()
    history.loc[0, "home_goals"] = 9
    with pytest.raises(ValueError, match="discrepan"):
        training_data(history, frame, "2026-09-16")


def test_snapshot_hash_detects_changes(tmp_path):
    target = tmp_path / "metadata.json"
    write_json(target, {"seed": 42})
    first = sha256(target)
    write_json(target, {"seed": 43})
    assert sha256(target) != first


def test_reject_invalid_round():
    with pytest.raises(ValueError, match="jornada"):
        as_of(fixture(), "2026-09-16", matchday=9)


def prepared_inputs(root):
    from champions.data import CURRENT_TEAMS
    teams = sorted(CURRENT_TEAMS)
    rows = []
    for day in range(8):
        for i in range(18):
            home, away = teams[i], teams[18 + (i + day) % 18]
            if day % 2:
                home, away = away, home
            rows.append(dict(match_id=f'{day}-{i}', date='2026-09-08T20:00:00Z' if day == 0 else f'2026-10-{day+1:02d}T20:00:00Z',
                             season='2026/27', stage='league', matchday=day + 1, home=home, away=away,
                             home_goals=1 if day == 0 else None, away_goals=0 if day == 0 else None,
                             status='finished' if day == 0 else 'scheduled', source='test'))
    fixtures = pd.DataFrame(rows)
    history = fixtures[['home', 'away']].copy()
    history['date'] = pd.date_range('2025-01-01', periods=len(history), tz='UTC')
    history['home_goals'], history['away_goals'] = 1, 0
    history['competition'], history['source'] = 'test', 'test'
    folder = root / 'data/processed'
    folder.mkdir(parents=True)
    fixtures.to_csv(folder / 'fixtures.csv', index=False)
    history.to_csv(folder / 'history.csv', index=False)
    return fixtures, history


def test_real_pipeline_persists_independent_runs_and_replays(tmp_path):
    import json
    from champions.pipeline import run, reproduce
    prepared_inputs(tmp_path)
    first = run(tmp_path, '2026-09-16', 1, simulations=100, evaluate_model=False, model_kind='baseline')
    second = run(tmp_path, '2026-09-16', 1, simulations=100, evaluate_model=False, model_kind='baseline')
    assert first != second and first.exists() and second.exists()
    pd.testing.assert_frame_equal(pd.read_csv(first / 'probabilities.csv'), pd.read_csv(second / 'probabilities.csv'))
    meta = json.loads((first / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['known_results'] == 18
    assert reproduce(tmp_path, first.name)['probabilities_reproduced']
    history_file = tmp_path / meta['data_snapshot'] / 'history.csv'
    history_file.write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='alterado'):
        reproduce(tmp_path, first.name)


def test_download_failure_does_not_publish_partial_generation(tmp_path, monkeypatch):
    from champions import data
    from champions.pipeline import update_data
    fixtures, history = prepared_inputs(tmp_path)
    monkeypatch.setattr(data, 'fetch_current', lambda raw: fixtures)
    monkeypatch.setattr(data, 'fetch_history', lambda raw: history)
    update_data(tmp_path)
    pointer = tmp_path / 'data/processed/current.json'
    previous = pointer.read_bytes()
    def fail(raw):
        raise ValueError('network source missing')
    monkeypatch.setattr(data, 'fetch_history', fail)
    with pytest.raises(ValueError, match='missing'):
        update_data(tmp_path)
    assert pointer.read_bytes() == previous
