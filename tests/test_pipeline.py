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
