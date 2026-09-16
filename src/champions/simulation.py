"""Batched UEFA 2026/27 league-phase and knockout Monte Carlo engine.

See docs/competition-rules.md for regulatory sources and modelling assumptions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = [
    "top8", "positions9_16", "positions17_24", "playoff", "eliminated",
    "round16", "quarterfinal", "semifinal", "final", "champion",
]


def _validate(fixtures):
    required = {"home", "away", "home_goals", "away_goals", "date", "matchday", "stage"}
    if not isinstance(fixtures, pd.DataFrame) or not required.issubset(fixtures.columns):
        raise ValueError(f"fixtures must contain columns {sorted(required)}")
    f = fixtures.copy(deep=True)
    stages = f.stage.astype(str).str.strip().str.lower().str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False)
    if not stages.isin(["league", "league_phase"]).all():
        raise NotImplementedError("Only league-phase fixtures are supported; supplied knockout/qualifying rows cannot be ignored.")
    for col in ("home", "away"):
        if not f[col].map(lambda x: isinstance(x, str) and bool(x.strip()) and x == x.strip()).all():
            raise ValueError("Team names must be nonempty strings without surrounding whitespace")
    teams = sorted(set(f.home) | set(f.away))
    if len(teams) != 36 or len(f) != 144:
        raise ValueError("A complete 36-club, 144-match league schedule is required")
    if (f.home == f.away).any():
        raise ValueError("A club cannot play itself")
    pairs = [tuple(sorted(pair)) for pair in zip(f.home, f.away)]
    if len(set(pairs)) != 144:
        raise ValueError("Duplicate opponents/fixtures in league schedule")
    for col in ("home", "away"):
        if not f[col].value_counts().reindex(teams, fill_value=0).eq(4).all():
            raise ValueError("Each club must have four home and four away fixtures")
    dates = pd.to_datetime(f.date, errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("Every fixture must have a valid date")
    days = pd.to_numeric(f.matchday, errors="coerce")
    if not days.isin(range(1, 9)).all():
        raise ValueError("matchday must be an integer from 1 to 8")
    f["matchday"] = days.astype(int)
    for _, day in f.groupby("matchday"):
        if len(day) != 18 or len(set(day.home) | set(day.away)) != 36:
            raise ValueError("Each matchday must contain every club exactly once")
    scores = f[["home_goals", "away_goals"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    known = ~np.isnan(scores)
    if (known[:, 0] != known[:, 1]).any():
        raise ValueError("Scores must be either both known or both missing")
    values = scores[known]
    if not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all():
        raise ValueError("Known scores must be finite nonnegative integers")
    f[["home_goals", "away_goals"]] = scores
    return f.sort_values(["matchday", "home", "away"]).reset_index(drop=True), teams


def _rank(metrics, rng):
    """Descending lexicographic ranking, with disclosed random unresolved ties."""
    keys = [rng.random(metrics.shape[:2])]
    keys.extend(-metrics[:, :, i] for i in range(metrics.shape[2]-1, -1, -1))
    order = np.lexsort(tuple(keys), axis=1)
    ordered = np.take_along_axis(metrics, order[:, :, None], axis=1)
    equal = (ordered[:, 1:] == ordered[:, :-1]).all(axis=2)
    starts = equal & ~np.pad(equal[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    return order, int(starts.sum()), int(equal.any(axis=1).sum())


def _league_metrics(hg, ag, incidence_h, incidence_a, opponents):
    """Nine available ranking keys, including full totals of all opponents."""
    hw, aw, draw = (hg > ag).astype(np.int64), (ag > hg).astype(np.int64), (hg == ag).astype(np.int64)
    points = (3*hw+draw) @ incidence_h + (3*aw+draw) @ incidence_a
    gf = hg @ incidence_h + ag @ incidence_a
    gd = gf - (ag @ incidence_h + hg @ incidence_a)
    return np.stack([points, gd, gf, ag @ incidence_a,
                     hw @ incidence_h + aw @ incidence_a, aw @ incidence_a,
                     points @ opponents, gd @ opponents, gf @ opponents], axis=2)


def _two_leg(first, second, rates, rng):
    """first hosts leg 1, second hosts leg 2 (including any extra time)."""
    leg1 = rates[0, first, second]
    leg2 = rates[0, second, first]
    a = rng.poisson(leg1[..., 0])
    b = rng.poisson(leg1[..., 1])
    b = b + rng.poisson(leg2[..., 0])
    a = a + rng.poisson(leg2[..., 1])
    tied = a == b
    # Sample extra-time scores only for tied aggregates, at one-third rates.
    b[tied] += rng.poisson(leg2[..., 0][tied] / 3)
    a[tied] += rng.poisson(leg2[..., 1][tied] / 3)
    a_wins = (a > b) | ((a == b) & (rng.random(a.shape) < .5))
    return np.where(a_wins, first, second)


def _final(first, second, rates, rng):
    lam = rates[1, first, second]
    a, b = rng.poisson(lam[..., 0]), rng.poisson(lam[..., 1])
    tied = a == b
    a[tied] += rng.poisson(lam[..., 0][tied] / 3)
    b[tied] += rng.poisson(lam[..., 1][tied] / 3)
    return np.where((a > b) | ((a == b) & (rng.random(a.shape) < .5)), first, second)


def _bracket(order, rates, rng):
    """Two sides, four lanes per side: rank-pair paths 1/2, 7/8, 3/4, 5/6."""
    size = len(order)

    def split_pairs(starts):
        swap = rng.integers(0, 2, size=(size, 4))
        slots = np.stack([np.array(starts) + swap, np.array(starts) + 1 - swap], axis=1)
        return order[np.arange(size)[:, None, None], slots]

    seeded = split_pairs([14, 8, 12, 10])
    unseeded = split_pairs([16, 22, 18, 20])
    po_winners = _two_leg(unseeded, seeded, rates, rng)
    r16_seeds = split_pairs([0, 6, 2, 4])
    qf = _two_leg(po_winners, r16_seeds, rates, rng)
    # Slot position, never the surviving club's original rank, owns priority.
    sf = _two_leg(qf[:, :, [1, 3]], qf[:, :, [0, 2]], rates, rng)
    finalists = _two_leg(sf[:, :, 1], sf[:, :, 0], rates, rng)
    champion = _final(finalists[:, 0], finalists[:, 1], rates, rng)
    return {
        "round16": np.concatenate([po_winners.reshape(size, 8), r16_seeds.reshape(size, 8)], axis=1),
        "quarterfinal": qf.reshape(size, 8), "semifinal": sf.reshape(size, 4),
        "final": finalists, "champion": champion[:, None],
    }


def simulate(fixtures: pd.DataFrame, model, simulations: int = 10000, seed: int = 42, knockout: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Forecast a complete league schedule and an undrawn knockout phase.

    model.predict_goals(home, away, neutral=False) must return two finite,
    nonnegative Poisson means. Non-league rows explicitly raise NotImplementedError.
    Missing disciplinary/coefficient criteria cause counted, warned random ties.
    No files are written; use probabilities.to_csv(..., index=False) externally.
    """
    if isinstance(simulations, (bool, np.bool_)) or not isinstance(simulations, (int, np.integer)) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    f, teams = _validate(fixtures)
    if knockout is not None:
        from champions.knockout import validate_state
        validate_state(knockout, teams, f)
    rng = np.random.default_rng(seed)
    index = {team: i for i, team in enumerate(teams)}
    home = f.home.map(index).to_numpy()
    away = f.away.map(index).to_numpy()
    rates = np.zeros((2, 36, 36, 2))
    for neutral in (False, True):
        for h, h_name in enumerate(teams):
            for a, a_name in enumerate(teams):
                if h == a:
                    continue
                value = np.asarray(model.predict_goals(h_name, a_name, neutral=neutral), dtype=float)
                if value.shape != (2,) or not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"Invalid predicted goal rates for {h_name} vs {a_name}")
                rates[int(neutral), h, a] = value
    incidence_h = np.eye(36, dtype=np.int64)[home]
    incidence_a = np.eye(36, dtype=np.int64)[away]
    opponents = incidence_h.T @ incidence_a + incidence_a.T @ incidence_h
    observed = f.home_goals.notna().to_numpy()
    unknown = ~observed
    counts = {col: np.zeros(36, dtype=np.int64) for col in PROBABILITY_COLUMNS}
    points_total = np.zeros(36, dtype=float)
    rank_total = np.zeros(36, dtype=float)
    unresolved_groups = unresolved_sims = 0
    for start in range(0, simulations, 1024):
        size = min(1024, simulations - start)
        hg = np.broadcast_to(f.home_goals.fillna(0).to_numpy(dtype=np.int64), (size, 144)).copy()
        ag = np.broadcast_to(f.away_goals.fillna(0).to_numpy(dtype=np.int64), (size, 144)).copy()
        hg[:, unknown] = rng.poisson(rates[0, home[unknown], away[unknown], 0], size=(size, int(unknown.sum())))
        ag[:, unknown] = rng.poisson(rates[0, home[unknown], away[unknown], 1], size=(size, int(unknown.sum())))
        metrics = _league_metrics(hg, ag, incidence_h, incidence_a, opponents)
        order, groups, tied_sims = _rank(metrics, rng)
        if knockout is not None:
            official = np.array([index[t] for t in knockout['league_order']])
            arranged = metrics[:, official]
            for j in range(35):
                if tuple(arranged[0,j]) < tuple(arranged[0,j+1]):
                    raise ValueError('La clasificación oficial contradice los resultados de fase liga.')
            order = np.broadcast_to(official, order.shape)
            groups = tied_sims = 0
        unresolved_groups += groups
        unresolved_sims += tied_sims
        selections = {"top8": order[:, :8], "positions9_16": order[:, 8:16],
                      "positions17_24": order[:, 16:24], "playoff": order[:, 8:24],
                      "eliminated": order[:, 24:]}
        if knockout is None:
            selections.update(_bracket(order, rates, rng))
        else:
            from champions.knockout import conditioned_bracket
            selections.update(conditioned_bracket(order, rates, rng, knockout, teams))
        for col, selected in selections.items():
            counts[col] += np.bincount(selected.ravel(), minlength=36)
        points_total += metrics[:, :, 0].sum(axis=0)
        rank_total += (np.argsort(order, axis=1) + 1).sum(axis=0)
    messages = []
    if unresolved_groups:
        messages.append(f"{unresolved_groups} unresolved league tie groups in {unresolved_sims}/{simulations} simulations: disciplinary points and club coefficients unavailable; seeded random ordering used, not an exact UEFA resolution.")
        warnings.warn(messages[0], RuntimeWarning, stacklevel=2)
    messages.extend([
        ("Knockout forecast assumes an undrawn bracket; supply a sourced knockout state for post-draw updates." if knockout is None else "Forecast conditioned on supplied official knockout draw, league order and completed legs."),
        "Disciplinary points and club coefficients are unavailable; unresolved ties, if any, use a counted random fallback.",
        "Goal simulation uses independent Poisson scores, constant rates, extra-time means divided by three, and 50/50 penalty shoot-outs.",
    ])
    probabilities = pd.DataFrame({"team": teams, **{col: counts[col]/simulations for col in PROBABILITY_COLUMNS},
                                  "expected_points": points_total/simulations, "expected_rank": rank_total/simulations})
    diagnostics = {
        "simulations": int(simulations), "seed": int(seed), "batch_size": 1024,
        "teams": 36, "league_matches": 144, "observed_league_matches": int(observed.sum()),
        "unplayed_league_matches": int(unknown.sum()), "model_predictions": 2520,
        "unresolved_tie_groups": unresolved_groups, "simulations_with_unresolved_ties": unresolved_sims,
        "unresolved_tie_simulation_fraction": unresolved_sims/simulations,
        "missing_tiebreakers": ["disciplinary_points", "club_coefficient"],
        "tie_fallback": "seeded_uniform_random_within_unresolved_group",
        "knockout_fixture_support": True, "knockout_conditioned": knockout is not None, "warnings": messages,
        "assumptions": ["independent_poisson_goals", "extra_time_rates_divided_by_3",
                        "penalty_shootout_50_50", "uniform_unconstrained_regulatory_draw",
                        "fixed_goal_rates_throughout_tournament"],
    }
    return probabilities, diagnostics
