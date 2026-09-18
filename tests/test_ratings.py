import pandas as pd
import pytest

from champions.ratings import (
    RatingConfig,
    build_rating_features,
    expected_score,
    goal_multiplier,
)


def _matches():
    return pd.DataFrame([
        {"date": "2026-08-01T12:00:00Z", "home": "A", "away": "B", "home_goals": 2,
         "away_goals": 0, "competition": "E0", "source": "test"},
        {"date": "2026-08-02T12:00:00Z", "home": "C", "away": "D", "home_goals": 0,
         "away_goals": 1, "competition": "N1", "source": "test"},
        {"date": "2026-08-10T12:00:00Z", "home": "A", "away": "C", "home_goals": 3,
         "away_goals": 0, "competition": "UCL", "source": "test"},
        {"date": "2026-09-01T12:00:00Z", "home": "B", "away": "D", "home_goals": 1,
         "away_goals": 0, "competition": "UCL", "source": "test"},
    ])


def test_expected_score_and_goal_multiplier_have_standard_elo_behavior():
    assert expected_score(1500, 1500, 0) == pytest.approx(0.5)
    assert expected_score(1550, 1500, 0) > 0.5
    assert goal_multiplier(0) == 1.0
    assert goal_multiplier(2) == 1.5
    assert goal_multiplier(4) > goal_multiplier(3)


def test_ratings_are_strictly_pre_match_and_future_results_do_not_leak_back():
    original = _matches()
    changed = original.copy()
    changed.loc[3, ["home_goals", "away_goals"]] = [9, 0]

    left = build_rating_features(original, "2026-10-01T00:00:00Z")
    right = build_rating_features(changed, "2026-10-01T00:00:00Z")

    pd.testing.assert_frame_equal(left.iloc[:-1], right.iloc[:-1])
    assert left.loc[0, "home_elo"] == pytest.approx(1500.0)
    assert left.loc[2, "home_elo"] > left.loc[0, "home_elo"]


def test_cross_league_results_update_regularized_offsets_and_recenter_them():
    result = build_rating_features(_matches(), "2026-10-01T00:00:00Z")
    offsets = result.attrs["league_offsets"]

    assert offsets["E0"] > offsets["N1"]
    assert sum(offsets.values()) == pytest.approx(0.0)
    assert result.attrs["team_leagues"] == {"A": "E0", "B": "E0", "C": "N1", "D": "N1"}


def test_new_club_in_a_rated_league_receives_the_league_offset_once():
    data = pd.concat([_matches().iloc[:3], pd.DataFrame([
        {"date": "2026-08-20T12:00:00Z", "home": "E", "away": "B", "home_goals": 1,
         "away_goals": 0, "competition": "E0", "source": "test"},
    ])], ignore_index=True)
    result = build_rating_features(data, "2026-10-01T00:00:00Z")
    debut = result.loc[result.home.eq("E")].iloc[0]
    assert debut.home_elo == pytest.approx(1500.0)
    assert debut.home_league_strength > 0


def test_replay_is_input_order_independent_and_retains_population_prior_in_metadata():
    data = _matches()
    a = build_rating_features(data, "2026-10-01T00:00:00Z")
    b = build_rating_features(data.sample(frac=1, random_state=7), "2026-10-01T00:00:00Z")

    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["config"]["initial"] == pytest.approx(1500.0)


@pytest.mark.parametrize("config", [
    RatingConfig(k_factor=0),
    RatingConfig(half_life_days=0),
    RatingConfig(home_advantage=float("inf")),
])
def test_invalid_rating_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        build_rating_features(_matches(), "2026-10-01T00:00:00Z", config)
