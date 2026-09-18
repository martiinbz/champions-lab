"""Chronological, leakage-safe team and league strength ratings."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


EUROPEAN_COMPETITIONS = frozenset({
    "UCL", "UCL qualifying", "UEL", "UEL qualifying", "UECL", "UECL qualifying",
})


@dataclass(frozen=True)
class RatingConfig:
    """Parameters for chronological Elo and cross-league strength updates."""

    initial: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 55.0
    half_life_days: float = 365.0
    league_k_factor: float = 8.0
    league_shrinkage: float = 20.0
    international_shrinkage: float = 8.0
    minimum_international_weight: float = 0.45

    def validate(self) -> None:
        finite = (self.initial, self.k_factor, self.home_advantage, self.half_life_days,
                  self.league_k_factor, self.league_shrinkage, self.international_shrinkage,
                  self.minimum_international_weight)
        if not np.isfinite(finite).all():
            raise ValueError("rating configuration values must be finite")
        if self.k_factor <= 0 or self.half_life_days <= 0:
            raise ValueError("k_factor and half_life_days must be positive")
        if self.league_k_factor < 0 or self.league_shrinkage <= 0 or self.international_shrinkage <= 0:
            raise ValueError("league parameters must be nonnegative with positive shrinkage")
        if not 0 <= self.minimum_international_weight <= 1:
            raise ValueError("minimum_international_weight must be within [0, 1]")


def expected_score(home_rating: float, away_rating: float, home_advantage: float) -> float:
    """Expected home result on the Elo 0-to-1 scale."""
    return float(1.0 / (1.0 + 10.0 ** (-(home_rating + home_advantage - away_rating) / 400.0)))


def goal_multiplier(goal_difference: int) -> float:
    """Moderate the Elo update by the 90-minute margin of victory."""
    if goal_difference < 0:
        raise ValueError("goal_difference cannot be negative")
    if goal_difference <= 1:
        return 1.0
    if goal_difference == 2:
        return 1.5
    return float((11 + goal_difference) / 8)


def is_cross_league_competition(competition: object) -> bool:
    return str(competition).strip() in EUROPEAN_COMPETITIONS


def _completed_history(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    required = {"date", "home", "away", "home_goals", "away_goals", "competition"}
    if not isinstance(history, pd.DataFrame) or not required.issubset(history.columns):
        raise ValueError(f"history must contain {sorted(required)}")
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce", format="mixed")
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    valid = (
        frame["date"].notna() & frame["date"].lt(cutoff) &
        frame["home"].map(lambda value: isinstance(value, str) and bool(value.strip())) &
        frame["away"].map(lambda value: isinstance(value, str) and bool(value.strip())) &
        frame["home"].ne(frame["away"]) &
        np.isfinite(frame["home_goals"]) & np.isfinite(frame["away_goals"]) &
        frame["home_goals"].ge(0) & frame["away_goals"].ge(0) &
        frame["home_goals"].eq(np.floor(frame["home_goals"])) &
        frame["away_goals"].eq(np.floor(frame["away_goals"]))
    )
    if not valid.all():
        raise ValueError("history contains invalid or incomplete match rows")
    return frame.loc[valid].sort_values(
        ["date", "competition", "home", "away"], kind="mergesort"
    ).reset_index(drop=True)


def build_rating_features(
    history: pd.DataFrame, cutoff: str, config: RatingConfig = RatingConfig(),
) -> pd.DataFrame:
    """Attach ratings known immediately before every historical match.

    Same-kickoff matches read one shared pre-match state, making results independent
    of input order and preventing a game on that timestamp influencing another.
    """
    config.validate()
    end = pd.Timestamp(cutoff)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if pd.isna(end):
        raise ValueError("cutoff must be a valid timestamp")
    frame = _completed_history(history, end)
    ratings: dict[str, float] = {}
    team_leagues: dict[str, str] = {}
    league_offsets: defaultdict[str, float] = defaultdict(float)
    league_evidence: defaultdict[str, float] = defaultdict(float)
    international_evidence: defaultdict[str, float] = defaultdict(float)
    records: list[dict] = []

    def rating_for(team: str, league: str | None) -> float:
        if team in ratings:
            return ratings[team]
        # Club ratings are residuals around the population mean. League strength
        # is injected separately in every expected-score calculation; adding it
        # here as well would count the same cross-league evidence twice.
        return config.initial

    for date, matches in frame.groupby("date", sort=True):
        snapshot_ratings = ratings.copy()
        snapshot_leagues = team_leagues.copy()
        snapshot_offsets = league_offsets.copy()
        snapshot_evidence = league_evidence.copy()
        snapshot_international = international_evidence.copy()
        updates: defaultdict[str, float] = defaultdict(float)
        league_updates: defaultdict[str, float] = defaultdict(float)
        domestic_assignments: dict[str, str] = {}
        days_old = max((end - date).total_seconds() / 86400.0, 0.0)
        recency = float(np.exp2(-days_old / config.half_life_days))

        def snapshot_rating(team: str, league: str | None) -> float:
            if team in snapshot_ratings:
                return snapshot_ratings[team]
            return config.initial

        def cross_league_rating(team: str, league: str | None) -> float:
            raw = snapshot_rating(team, league)
            evidence = snapshot_international[team]
            weight = config.minimum_international_weight + (1.0 - config.minimum_international_weight) * evidence / (evidence + config.international_shrinkage)
            return config.initial + (raw - config.initial) * weight

        for row in matches.itertuples(index=False):
            cross_league = is_cross_league_competition(row.competition)
            home_league = snapshot_leagues.get(row.home) if cross_league else str(row.competition)
            away_league = snapshot_leagues.get(row.away) if cross_league else str(row.competition)
            home_rating = cross_league_rating(row.home, home_league) if cross_league else snapshot_rating(row.home, home_league)
            away_rating = cross_league_rating(row.away, away_league) if cross_league else snapshot_rating(row.away, away_league)
            home_offset = snapshot_offsets.get(home_league, 0.0) if home_league else 0.0
            away_offset = snapshot_offsets.get(away_league, 0.0) if away_league else 0.0
            expected = expected_score(home_rating + home_offset, away_rating + away_offset, config.home_advantage)
            actual = 1.0 if row.home_goals > row.away_goals else 0.5 if row.home_goals == row.away_goals else 0.0
            delta = config.k_factor * recency * goal_multiplier(int(abs(row.home_goals - row.away_goals))) * (actual - expected)
            updates[row.home] += delta
            updates[row.away] -= delta
            records.append({
                **row._asdict(),
                "home_elo": home_rating,
                "away_elo": away_rating,
                "home_league": home_league,
                "away_league": away_league,
                "home_league_strength": home_offset,
                "away_league_strength": away_offset,
                "elo_expected_home": expected,
                "elo_recency_weight": recency,
            })
            if cross_league and home_league and away_league and home_league != away_league:
                league_delta = config.league_k_factor * recency * (actual - expected)
                league_updates[home_league] += league_delta
                league_updates[away_league] -= league_delta
                league_evidence[home_league] += recency
                league_evidence[away_league] += recency
                international_evidence[row.home] += recency
                international_evidence[row.away] += recency
            if not cross_league:
                domestic_assignments[row.home] = str(row.competition)
                domestic_assignments[row.away] = str(row.competition)

        for team, delta in updates.items():
            league = domestic_assignments.get(team, team_leagues.get(team))
            ratings[team] = rating_for(team, league) + delta
        team_leagues.update(domestic_assignments)
        for league, delta in league_updates.items():
            league_offsets[league] += delta
        if league_offsets:
            weighted_mean = sum(league_offsets[name] * league_evidence[name] for name in league_offsets)
            total_weight = sum(league_evidence.values())
            if total_weight:
                for league in league_offsets:
                    league_offsets[league] -= weighted_mean / total_weight

    result = pd.DataFrame(records)
    result.attrs["config"] = asdict(config)
    final_ratings = {}
    for team, value in ratings.items():
        evidence = international_evidence[team]
        weight = config.minimum_international_weight + (1.0 - config.minimum_international_weight) * evidence / (evidence + config.international_shrinkage)
        final_ratings[team] = float(config.initial + (value - config.initial) * weight)
    result.attrs["final_ratings"] = dict(sorted(final_ratings.items()))
    result.attrs["league_offsets"] = dict(sorted((league, float(value)) for league, value in league_offsets.items()))
    result.attrs["league_evidence"] = dict(sorted((league, float(value)) for league, value in league_evidence.items()))
    result.attrs["international_evidence"] = dict(sorted((team, float(value)) for team, value in international_evidence.items()))
    result.attrs["team_leagues"] = dict(sorted(team_leagues.items()))
    return result
