"""Dixon-Coles score model enriched with chronological Elo and league ratings."""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from champions.models import GoalModel, independent_poisson_grid
from champions.ratings import RatingConfig, build_rating_features


def tau(home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float) -> float:
    """Dixon-Coles correction for the four dependent low-score outcomes."""
    if (home_goals, away_goals) == (0, 0):
        return 1.0 - home_rate * away_rate * rho
    if (home_goals, away_goals) == (0, 1):
        return 1.0 + home_rate * rho
    if (home_goals, away_goals) == (1, 0):
        return 1.0 + away_rate * rho
    if (home_goals, away_goals) == (1, 1):
        return 1.0 - rho
    return 1.0


def dixon_coles_grid(home_rate: float, away_rate: float, rho: float, max_goals: int = 12) -> np.ndarray:
    """Return a normalized independent-Poisson grid with low-score correction."""
    grid = independent_poisson_grid(home_rate, away_rate, max_goals)
    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        grid[home_goals, away_goals] *= tau(home_goals, away_goals, home_rate, away_rate, rho)
    if not np.isfinite(grid).all() or (grid < 0).any() or grid.sum() <= 0:
        raise ValueError("Dixon-Coles correction creates an invalid score distribution")
    return grid / grid.sum()


class DixonColesModel(GoalModel):
    """Regularized Poisson strengths plus pre-match Elo and Dixon-Coles dependence."""

    def __init__(self, *, half_life_days: float = 180.0, regularization: float = 8.0,
                 prior_goals: float = 1.35, prior_home_advantage: float = 0.15,
                 prior_matches: float = 10.0, min_team_matches: int = 5,
                 max_iter: int = 500, tolerance: float = 1e-9,
                 elo_scale: float = 0.0015, league_scale: float = 0.0015,
                 rho_bound: float = 0.2, rating_config: RatingConfig = RatingConfig()):
        super().__init__(kind="poisson", half_life_days=half_life_days,
                         regularization=regularization, prior_goals=prior_goals,
                         prior_home_advantage=prior_home_advantage,
                         prior_matches=prior_matches, min_team_matches=min_team_matches,
                         max_iter=max_iter, tolerance=tolerance)
        for name, value in (("elo_scale", elo_scale), ("league_scale", league_scale), ("rho_bound", rho_bound)):
            if not np.isfinite(value) or (name == "rho_bound" and not 0 < value < 1) or (
                name != "rho_bound" and value < 0
            ):
                raise ValueError(f"{name} must be finite and valid")
        rating_config.validate()
        self.params.update(elo_scale=float(elo_scale), league_scale=float(league_scale),
                           rho_bound=float(rho_bound), rating_config=copy.deepcopy(rating_config.__dict__))
        self.elo_coefficient = float(elo_scale)
        self.league_coefficient = float(league_scale)
        self.rho = 0.0
        self.rating_config = rating_config
        self.final_ratings: dict[str, float] = {}
        self.league_offsets: dict[str, float] = {}
        self.team_leagues: dict[str, str] = {}

    def _adjusted_rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        base_home, base_away = super().predict_goals(home, away, neutral=neutral)
        home_elo = self.final_ratings.get(home, self.rating_config.initial)
        away_elo = self.final_ratings.get(away, self.rating_config.initial)
        home_league = self.team_leagues.get(home)
        away_league = self.team_leagues.get(away)
        league_difference = self.league_offsets.get(home_league, 0.0) - self.league_offsets.get(away_league, 0.0)
        adjustment = self.elo_coefficient * (home_elo - away_elo) + self.league_coefficient * league_difference
        values = np.exp(np.log([base_home, base_away]) + [adjustment, -adjustment])
        return tuple(map(float, np.clip(values, 1e-6, 100.0)))

    def fit(self, history: pd.DataFrame, cutoff: str) -> "DixonColesModel":
        super().fit(history, cutoff)
        features = build_rating_features(history, cutoff, self.rating_config)
        self.final_ratings = features.attrs["final_ratings"]
        self.league_offsets = features.attrs["league_offsets"]
        self.team_leagues = features.attrs["team_leagues"]
        base_fit_info = copy.deepcopy(self.fit_info)
        if features.empty or not base_fit_info.get("success"):
            self.fit_info = {**base_fit_info, "method": "Dixon-Coles", "fallback": "poisson",
                             "reason": "no_features_or_poisson_fallback",
                             "rating_config": copy.deepcopy(self.rating_config.__dict__)}
            return self

        home_rates, away_rates = [], []
        for row in features.itertuples(index=False):
            home_rate, away_rate = GoalModel.predict_goals(self, row.home, row.away)
            home_rates.append(home_rate)
            away_rates.append(away_rate)
        base_home = np.asarray(home_rates)
        base_away = np.asarray(away_rates)
        elo_difference = features.home_elo.to_numpy(dtype=float) - features.away_elo.to_numpy(dtype=float)
        league_difference = features.home_league_strength.to_numpy(dtype=float) - features.away_league_strength.to_numpy(dtype=float)
        home_goals = features.home_goals.to_numpy(dtype=int)
        away_goals = features.away_goals.to_numpy(dtype=int)
        ages = (pd.Timestamp(cutoff, tz="UTC") - pd.to_datetime(features.date, utc=True)).dt.total_seconds().to_numpy() / 86400.0
        weights = np.exp2(-np.maximum(ages, 0.0) / self.params["half_life_days"])

        def objective(values: np.ndarray) -> float:
            elo_coefficient, league_coefficient, rho = values
            adjustment = elo_coefficient * elo_difference + league_coefficient * league_difference
            lambda_home = base_home * np.exp(np.clip(adjustment, -5, 5))
            lambda_away = base_away * np.exp(np.clip(-adjustment, -5, 5))
            correction = np.ones(len(features))
            for score, formula in {
                (0, 0): 1.0 - lambda_home * lambda_away * rho,
                (0, 1): 1.0 + lambda_home * rho,
                (1, 0): 1.0 + lambda_away * rho,
                (1, 1): 1.0 - rho,
            }.items():
                mask = (home_goals == score[0]) & (away_goals == score[1])
                correction[mask] = np.asarray(formula)[mask] if np.ndim(formula) else formula
            if (correction <= 0).any() or not np.isfinite(correction).all():
                return float("inf")
            log_likelihood = poisson.logpmf(home_goals, lambda_home) + poisson.logpmf(away_goals, lambda_away) + np.log(correction)
            penalty = self.params["regularization"] * (elo_coefficient ** 2 + league_coefficient ** 2 + rho ** 2)
            return float(-np.dot(weights, log_likelihood) + penalty)

        result = minimize(objective, [self.elo_coefficient, self.league_coefficient, 0.0], method="L-BFGS-B",
                          bounds=[(-0.01, 0.01), (-0.01, 0.01), (-self.params["rho_bound"], self.params["rho_bound"])],
                          options={"maxiter": self.params["max_iter"], "ftol": self.params["tolerance"]})
        success = bool(result.success and np.isfinite(result.fun) and np.all(np.isfinite(result.x)))
        if success:
            self.elo_coefficient, self.league_coefficient, self.rho = map(float, result.x)
        else:
            self.elo_coefficient = self.league_coefficient = self.rho = 0.0
        self.fit_info = {
            **base_fit_info, "method": "Dixon-Coles L-BFGS-B", "success": success,
            "iterations": int(result.nit), "objective": float(result.fun) if np.isfinite(result.fun) else None,
            "message": str(result.message), "fallback": None if success else "poisson",
            "rating_config": copy.deepcopy(self.rating_config.__dict__),
        }
        if not success:
            self._warn("Dixon-Coles optimization did not converge; using Poisson score dependence.", RuntimeWarning)
        return self

    def predict_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        return self._adjusted_rates(home, away, neutral)

    def score_probabilities(self, home: str, away: str, neutral: bool = False,
                            max_goals: int = 12) -> np.ndarray:
        return dixon_coles_grid(*self.predict_goals(home, away, neutral), self.rho, max_goals)

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload.update(rho=self.rho, elo_coefficient=self.elo_coefficient,
                       league_coefficient=self.league_coefficient,
                       final_ratings=copy.deepcopy(self.final_ratings),
                       league_offsets=copy.deepcopy(self.league_offsets),
                       team_leagues=copy.deepcopy(self.team_leagues))
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "DixonColesModel":
        data = copy.deepcopy(payload)
        if data.get("schema_version") != 1 or data.get("model_type") != cls.__name__:
            raise ValueError("unsupported schema_version or incorrect model_type")
        params = data.get("params", {})
        required = {"rho", "elo_coefficient", "league_coefficient", "final_ratings", "league_offsets", "team_leagues"}
        if not required.issubset(data) or "rating_config" not in params:
            raise ValueError("incomplete Dixon-Coles serialization")
        allowed = {"half_life_days", "regularization", "prior_goals", "prior_home_advantage", "prior_matches",
                   "min_team_matches", "max_iter", "tolerance", "elo_scale", "league_scale", "rho_bound"}
        model = cls(**{key: params[key] for key in allowed if key in params},
                    rating_config=RatingConfig(**params["rating_config"]))
        model.params = params
        numeric = [data["intercept"], data["home_advantage"], data["weighted_matches"], data["rho"],
                   data["elo_coefficient"], data["league_coefficient"], *data["attack"].values(),
                   *data["defence"].values(), *data["final_ratings"].values(), *data["league_offsets"].values()]
        if not np.isfinite(numeric).all() or not set(data["attack"]) == set(data["defence"]):
            raise ValueError("invalid Dixon-Coles numerical payload")
        for key in ("train_cutoff", "train_last_date", "n_matches", "weighted_matches", "coverage", "intercept",
                    "home_advantage", "attack", "defence", "fit_info", "warnings", "rho", "elo_coefficient",
                    "league_coefficient", "final_ratings", "league_offsets", "team_leagues"):
            setattr(model, key, data[key])
        model._fitted = True
        return model
