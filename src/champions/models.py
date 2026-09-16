"""Regularized independent-Poisson goal models and chronological evaluation.

Only observed scores strictly before the cutoff enter estimation. See docs/models.md
for the completion-time contract, probability semantics, and evaluation limitations.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import skellam

__all__ = ["GoalModel", "BaselineModel", "CoverageWarning", "DataWarning",
           "outcome_probabilities", "evaluate"]


class CoverageWarning(UserWarning):
    """Missing or sparse team evidence; predictions depend on priors."""


class DataWarning(UserWarning):
    """Invalid or incomplete historical observations were excluded."""


def _cutoff(value):
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("missing timestamp")
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    except (ValueError, TypeError) as exc:
        raise ValueError("cutoff must be a valid timestamp") from exc


def _completed(history, cutoff):
    required = ["date", "home", "away", "home_goals", "away_goals"]
    if not isinstance(history, pd.DataFrame) or not set(required).issubset(history.columns):
        raise ValueError(f"history must contain {required}")
    if history.columns.duplicated().any():
        raise ValueError("history has duplicate column names")
    frame = history[required + (["completed_at"] if "completed_at" in history else [])].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce", utc=True, format="mixed")
    # Do not inspect scores or team identities of future observations.
    frame = frame.loc[frame.date < cutoff].copy()
    if "completed_at" in frame:
        frame["completed_at"] = pd.to_datetime(frame.completed_at, errors="coerce", utc=True, format="mixed")
        frame = frame.loc[(frame.completed_at < cutoff) & (frame.completed_at >= frame.date)].copy()
    valid = pd.Series(True, index=frame.index)
    for column in ("home", "away"):
        valid &= frame[column].map(lambda x: isinstance(x, str) and bool(x.strip()))
    valid &= frame.home != frame.away
    for column in ("home_goals", "away_goals"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        valid &= np.isfinite(values) & (values >= 0) & (values == np.floor(values))
    if not valid.all():
        warnings.warn(f"Excluded {int((~valid).sum())} incomplete or invalid pre-cutoff matches.",
                      DataWarning, stacklevel=3)
    # Canonical ordering makes fitting and serialization insensitive to input order.
    return frame.loc[valid].sort_values(required, kind="mergesort").reset_index(drop=True)


def outcome_probabilities(home_goals: float, away_goals: float) -> dict[str, float]:
    """Return home/draw/away probabilities, integrating all Poisson score tails."""
    rates = np.asarray([home_goals, away_goals], dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("goal rates must be finite and positive")
    p = np.array([skellam.sf(0, *rates), skellam.pmf(0, *rates), skellam.cdf(-1, *rates)])
    if not np.all(np.isfinite(p)) or p.sum() <= 0:
        raise ValueError("goal rates exceed numerical probability support")
    p = np.maximum(p, 0)
    p /= p.sum()
    return dict(zip(("home", "draw", "away"), map(float, p)))


class BaselineModel:
    """Time-decayed league scoring rates with prior pseudo-match smoothing."""

    def __init__(self, *, half_life_days=365.0, prior_goals=1.35,
                 prior_home_advantage=0.15, prior_matches=10.0, min_team_matches=5):
        for name, value in (("half_life_days", half_life_days), ("prior_goals", prior_goals),
                            ("prior_matches", prior_matches)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(prior_home_advantage) or abs(prior_home_advantage) > 1.5:
            raise ValueError("prior_home_advantage must be finite and within [-1.5, 1.5]")
        if not isinstance(min_team_matches, int) or min_team_matches < 1:
            raise ValueError("min_team_matches must be a positive integer")
        self.params = dict(half_life_days=float(half_life_days), prior_goals=float(prior_goals),
                           prior_home_advantage=float(prior_home_advantage),
                           prior_matches=float(prior_matches), min_team_matches=min_team_matches)
        self._fitted = False
        self.warnings = []

    def _warn(self, message, category=CoverageWarning):
        if message not in self.warnings:
            self.warnings.append(message)
        warnings.warn(message, category, stacklevel=3)

    def fit(self, history: pd.DataFrame, cutoff: str) -> BaselineModel:
        cutoff = _cutoff(cutoff)
        frame = _completed(history, cutoff)
        self._fitted = False
        self.warnings = []
        age = (cutoff - frame.date).dt.total_seconds().to_numpy() / 86400
        weights = np.exp2(-age / self.params["half_life_days"])
        self.train_cutoff = cutoff.isoformat()
        self.n_matches = len(frame)
        self.weighted_matches = float(weights.sum())
        self.train_last_date = frame.date.max().isoformat() if len(frame) else None
        self.coverage = {}
        for team in sorted(set(frame.home) | set(frame.away)):
            mask = ((frame.home == team) | (frame.away == team)).to_numpy()
            self.coverage[team] = {"matches": int(mask.sum()),
                                   "weighted_matches": float(weights[mask].sum())}
        sparse = [team for team, c in self.coverage.items()
                  if c["weighted_matches"] < self.params["min_team_matches"]]
        if not len(frame) or sparse:
            self._warn("Limited training coverage; priors dominate for " +
                       ((", ".join(sparse[:12]) + (f" and {len(sparse)-12} other teams (see model coverage)." if len(sparse) > 12 else ""))
                        if sparse else "all teams (no completed matches)"))
        prior = self.params["prior_matches"]
        away = (np.dot(weights, frame.away_goals.to_numpy(dtype=float)) + prior * self.params["prior_goals"]) / (weights.sum() + prior)
        home = (np.dot(weights, frame.home_goals.to_numpy(dtype=float)) + prior * self.params["prior_goals"] * np.exp(self.params["prior_home_advantage"])) / (weights.sum() + prior)
        self.intercept = float(np.log(away))
        self.home_advantage = float(np.log(home / away))
        self.attack = {team: 0.0 for team in self.coverage}
        self.defence = {team: 0.0 for team in self.coverage}
        self.fit_info = {"method": "smoothed_decayed_rates", "success": True}
        self._fitted = True
        return self

    def _check_coverage(self, home, away):
        if not self._fitted:
            raise RuntimeError("fit the model before prediction or serialization")
        for team in (home, away):
            if not isinstance(team, str) or not team.strip():
                raise ValueError("team names must be nonempty strings")
            count = self.coverage.get(team)
            if count is None:
                self._warn(f"Missing coverage for {team!r}; using zero team strength (league prior).")
            elif count["weighted_matches"] < self.params["min_team_matches"]:
                self._warn(f"Sparse coverage for {team!r}: {count['matches']} matches, "
                           f"{count['weighted_matches']:.2f} decayed matches; prior-sensitive prediction.")

    def predict_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        self._check_coverage(home, away)
        h = self.intercept + self.attack.get(home, 0) - self.defence.get(away, 0)
        a = self.intercept + self.attack.get(away, 0) - self.defence.get(home, 0)
        if not neutral:
            h += self.home_advantage
        return tuple(map(float, np.exp(np.clip([h, a], np.log(1e-6), np.log(100)))))

    def predict_outcomes(self, home: str, away: str, neutral: bool = False) -> dict[str, float]:
        return outcome_probabilities(*self.predict_goals(home, away, neutral))

    def to_dict(self) -> dict:
        if not self._fitted:
            raise RuntimeError("fit the model before serialization")
        return copy.deepcopy(dict(schema_version=1, model_type=type(self).__name__,
                                  params=self.params, train_cutoff=self.train_cutoff,
                                  train_last_date=self.train_last_date, n_matches=self.n_matches,
                                  weighted_matches=self.weighted_matches, coverage=self.coverage,
                                  intercept=self.intercept, home_advantage=self.home_advantage,
                                  attack=self.attack, defence=self.defence, fit_info=self.fit_info,
                                  warnings=self.warnings))

    @classmethod
    def from_dict(cls, payload: dict) -> BaselineModel:
        data = copy.deepcopy(payload)
        if data.get("schema_version") != 1 or data.get("model_type") != cls.__name__:
            raise ValueError("unsupported schema_version or incorrect model_type")
        model = cls(**data["params"])
        _cutoff(data["train_cutoff"])
        names = set(data["coverage"])
        if set(data["attack"]) != names or set(data["defence"]) != names:
            raise ValueError("strength and coverage teams must agree")
        numeric = [data["intercept"], data["home_advantage"], data["weighted_matches"],
                   *data["attack"].values(), *data["defence"].values()]
        numeric += [c["weighted_matches"] for c in data["coverage"].values()]
        if not np.all(np.isfinite(numeric)) or data["weighted_matches"] < 0:
            raise ValueError("model contains invalid numerical parameters")
        if not isinstance(data["n_matches"], int) or data["n_matches"] < 0:
            raise ValueError("invalid match count")
        if data["train_last_date"] is not None and _cutoff(data["train_last_date"]) >= _cutoff(data["train_cutoff"]):
            raise ValueError("training date must precede cutoff")
        for key in ("train_cutoff", "train_last_date", "n_matches", "weighted_matches", "coverage",
                    "intercept", "home_advantage", "attack", "defence", "fit_info"):
            setattr(model, key, data[key])
        model.warnings = data.get("warnings", [])
        model._fitted = True
        return model


class GoalModel(BaselineModel):
    """MAP Poisson attack/defence model with exponential decay and L2 priors."""

    def __init__(self, *, kind="poisson", half_life_days=365.0, regularization=5.0, prior_goals=1.35,
                 prior_home_advantage=0.15, prior_matches=10.0, min_team_matches=5,
                 max_iter=500, tolerance=1e-9):
        super().__init__(half_life_days=half_life_days, prior_goals=prior_goals,
                         prior_home_advantage=prior_home_advantage,
                         prior_matches=prior_matches, min_team_matches=min_team_matches)
        if kind not in ("poisson", "baseline"):
            raise ValueError("kind must be 'poisson' or 'baseline'")
        self.kind = kind
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.params.update(kind=kind, regularization=float(regularization), max_iter=max_iter,
                           tolerance=float(tolerance))

    def fit(self, history: pd.DataFrame, cutoff: str) -> GoalModel:
        # Baseline initializes an explicit fallback if optimization fails.
        super().fit(history, cutoff)
        if self.kind == "baseline":
            return self
        frame = _completed(history, _cutoff(cutoff))
        if frame.empty:
            self.fit_info = {"method": "prior_only", "success": False,
                             "fallback": "baseline", "reason": "no_completed_matches"}
            return self
        teams = list(self.coverage)
        index = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        h = frame.home.map(index).to_numpy()
        a = frame.away.map(index).to_numpy()
        yh = frame.home_goals.to_numpy(dtype=float)
        ya = frame.away_goals.to_numpy(dtype=float)
        age = (_cutoff(cutoff) - frame.date).dt.total_seconds().to_numpy() / 86400
        weights = np.exp2(-age / self.params["half_life_days"])
        prior_mean = np.zeros(2 + 2 * n)
        prior_mean[:2] = [np.log(self.params["prior_goals"]), self.params["prior_home_advantage"]]
        penalty = np.full(2 + 2 * n, self.params["regularization"])
        penalty[:2] = self.params["prior_matches"]
        initial = np.zeros_like(prior_mean)
        initial[:2] = [self.intercept, self.home_advantage]

        def objective(x):
            attack, defence = x[2:2+n], x[2+n:]
            lh = x[0] + x[1] + attack[h] - defence[a]
            la = x[0] + attack[a] - defence[h]
            mh, ma = np.exp(lh), np.exp(la)
            delta = x - prior_mean
            value = np.dot(weights, mh - yh * lh + ma - ya * la) + .5 * np.dot(penalty, delta * delta)
            rh, ra = weights * (mh - yh), weights * (ma - ya)
            grad = penalty * delta
            grad[0] += (rh + ra).sum()
            grad[1] += rh.sum()
            grad[2:2+n] += np.bincount(h, rh, n) + np.bincount(a, ra, n)
            grad[2+n:] -= np.bincount(a, rh, n) + np.bincount(h, ra, n)
            return float(value), grad

        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=[(-3, 3), (-1.5, 1.5)] + [(-2.5, 2.5)] * (2 * n),
                          options={"maxiter": self.params["max_iter"], "ftol": self.params["tolerance"]})
        success = bool(result.success and np.isfinite(result.fun) and np.all(np.isfinite(result.x)))
        self.fit_info = {"method": "L-BFGS-B", "success": success,
                         "iterations": int(result.nit), "message": str(result.message),
                         "objective": float(result.fun) if np.isfinite(result.fun) else None,
                         "fallback": None if success else "baseline"}
        if success:
            self.intercept, self.home_advantage = map(float, result.x[:2])
            self.attack = dict(zip(teams, map(float, result.x[2:2+n])))
            self.defence = dict(zip(teams, map(float, result.x[2+n:])))
        else:
            self._warn("Poisson optimization did not converge; using smoothed baseline rates.", RuntimeWarning)
        return self


def _metrics(model, holdout, bins):
    # Aggregate missing coverage in the report and emit a single warning below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoverageWarning)
        rates = np.array([model.predict_goals(row.home, row.away) for row in holdout.itertuples()])
    probabilities = np.array([list(outcome_probabilities(*r).values()) for r in rates])
    goals = holdout[["home_goals", "away_goals"]].to_numpy(dtype=float)
    labels = np.where(goals[:, 0] > goals[:, 1], 0, np.where(goals[:, 0] == goals[:, 1], 1, 2))
    targets = np.eye(3)[labels]
    calibration = {}
    for column, name in enumerate(("home", "draw", "away")):
        assignments = np.minimum((probabilities[:, column] * bins).astype(int), bins - 1)
        calibration[name] = []
        for i in range(bins):
            mask = assignments == i
            count = int(mask.sum())
            calibration[name].append({"lower": i / bins, "upper": (i + 1) / bins,
                                      "count": count,
                                      "mean_probability": float(probabilities[mask, column].mean()) if count else None,
                                      "observed_frequency": float(targets[mask, column].mean()) if count else None})
    unknown = sorted((set(holdout.home) | set(holdout.away)) - set(model.coverage))
    sparse = sorted(team for team in (set(holdout.home) | set(holdout.away)) & set(model.coverage)
                    if model.coverage[team]["weighted_matches"] < model.params["min_team_matches"])
    if unknown or sparse:
        warnings.warn(f"Holdout has missing coverage {unknown} and sparse coverage {sparse}.",
                      CoverageWarning, stacklevel=3)
    return {"log_loss": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-15)).mean()),
            "brier": float(np.square(probabilities - targets).sum(axis=1).mean()),
            "goal_log_loss": float((rates - goals * np.log(rates) + gammaln(goals + 1)).sum(axis=1).mean()),
            "calibration_bins": calibration, "missing_teams": unknown, "sparse_teams": sparse,
            "fit_info": copy.deepcopy(model.fit_info)}


def evaluate(history: pd.DataFrame, cutoff: str, *, holdout_fraction: float = .2,
             min_train: int = 30, min_holdout: int = 10, max_holdout: int = 2000,
             calibration_bins: int = 10, model_params: dict | None = None) -> dict:
    """Fit once per candidate before a fixed temporal holdout, all before cutoff.

    Recommend on 1X2 log loss; this selection holdout is not an unbiased test set.
    Evaluation never changes a fitted model or fits on its held-out observations.
    """
    if not np.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    for name, value in (("min_train", min_train), ("min_holdout", min_holdout),
                        ("max_holdout", max_holdout), ("calibration_bins", calibration_bins)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_holdout < min_holdout:
        raise ValueError("max_holdout must be at least min_holdout")
    cutoff_stamp = _cutoff(cutoff)
    frame = _completed(history, cutoff_stamp)
    report = {"status": "insufficient_data", "cutoff": cutoff_stamp.isoformat(),
              "n_available": len(frame), "n_train": 0, "n_holdout": 0,
              "recommended_model": "baseline", "selected_model": "baseline", "models": {},
              "selection_metric": "log_loss", "holdout_truncated": False,
              "limitation": "A single selection holdout does not establish reliability or unbiased future performance."}
    if len(frame) < min_train + min_holdout:
        report["reason"] = "too_few_completed_matches"
        return report
    split_index = min(len(frame) - min_holdout, max(min_train, int(len(frame) * (1 - holdout_fraction))))
    split_date = frame.iloc[split_index].date
    # Keep every identical timestamp on the same side; enforce result availability.
    train = _completed(frame.loc[frame.date < split_date], split_date)
    holdout_all = frame.loc[frame.date >= split_date]
    # Earliest observations preserve a bounded, contiguous horizon after the split.
    holdout = holdout_all.iloc[:max_holdout]
    report.update(n_train=len(train), n_holdout=len(holdout),
                  holdout_truncated=len(holdout_all) > max_holdout)
    if len(train) < min_train or len(holdout) < min_holdout:
        report["reason"] = "too_few_matches_after_timestamp_grouping"
        return report
    candidate = GoalModel(**(model_params or {}))
    if candidate.kind != "poisson":
        raise ValueError("evaluate compares Poisson to baseline; model_params.kind must be 'poisson'")
    base_keys = {"half_life_days", "prior_goals", "prior_home_advantage", "prior_matches", "min_team_matches"}
    baseline = BaselineModel(**{k: v for k, v in candidate.params.items() if k in base_keys})
    for name, model in (("poisson", candidate), ("baseline", baseline)):
        model.fit(train, split_date.isoformat())
        report["models"][name] = _metrics(model, holdout, calibration_bins)
    report.update(status="ok", train_cutoff=split_date.isoformat(),
                  train_last_date=train.date.max().isoformat(),
                  holdout_start=holdout.date.min().isoformat(),
                  holdout_end=holdout.date.max().isoformat(), model_params=candidate.params.copy())
    if candidate.fit_info["success"] and report["models"]["poisson"]["log_loss"] < report["models"]["baseline"]["log_loss"]:
        report["recommended_model"] = "poisson"
    report["selected_model"] = report["recommended_model"]
    return report
