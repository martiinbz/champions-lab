"""Competition-specific expanding-window validation with a reserved final block."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from champions.models import GoalModel, BaselineModel, CoverageWarning, _completed, _cutoff, _metrics, evaluate
from champions.dixon_coles import DixonColesModel
from champions.ensemble import EnsembleModel
from champions.calibration import CalibratedModel, OutcomeCalibrator, grid_outcomes


def _outcome_records(model, heldout: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return only pre-match 1X2 probabilities and realised result indices."""
    probabilities, outcomes = [], []
    for row in heldout.itertuples(index=False):
        probabilities.append(grid_outcomes(model.score_probabilities(row.home, row.away)))
        outcomes.append(0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2)
    return np.asarray(probabilities), np.asarray(outcomes, dtype=int)


def _outcome_metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    return {"log_loss": OutcomeCalibrator.log_loss(probabilities, outcomes),
            "brier": float(np.mean(np.sum((probabilities - np.eye(3)[outcomes]) ** 2, axis=1)))}


def _candidate(name: str):
    if name == "poisson":
        return GoalModel()
    if name == "baseline":
        return BaselineModel()
    if name == "dixon_coles":
        return DixonColesModel(max_iter=150)
    raise ValueError("unsupported calibration base model")


def evaluate_rolling(history: pd.DataFrame, cutoff: str, folds: int = 3) -> dict:
    """Use only UCL holdouts; all competitions may supply pre-block training.

    First folds select the model. The final block is reserved for assessment and
    never changes that selection. Each block refits using strictly earlier data.
    """
    if folds < 2:
        raise ValueError("At least two folds are needed for selection and assessment.")
    report = {"status": "insufficient_data", "folds": [], "selection_folds": folds - 1,
              "assessment_fold": folds, "competition": "UCL",
              "promotion": {"minimum_log_loss_improvement": .005, "minimum_selection_folds": 2},
              "calibration": {"status": "insufficient_data", "promoted": False, "reason": "insufficient_data", "payload": None}}
    if "competition" not in history:
        report["reason"] = "competition_labels_unavailable"
        return report
    cut = _cutoff(cutoff)
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    ucl = _completed(frame.loc[frame.competition.eq("UCL")], cut)
    dates = ucl.date.drop_duplicates().sort_values().to_list()
    if len(ucl) < 80 or len(dates) < 4 * folds:
        report["reason"] = "too_few_UCL_matches_or_dates"
        return report
    boundaries = [dates[int(len(dates) * (i + 1) / (folds + 1))] for i in range(folds)] + [cut]
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        # Keep competition labels for chronological league-strength features.
        # Each candidate still applies its own strict pre-cutoff score validation.
        train = frame.loc[frame.date.lt(start)].copy()
        heldout = ucl.loc[ucl.date.ge(start) & ucl.date.lt(end)]
        if len(train) < 100 or len(heldout) < 10:
            report["reason"] = "insufficient_fold_coverage"
            return report
        fold = {"fold": i + 1, "purpose": "assessment" if i == folds - 1 else "selection",
                "train_last_date": train.date.max().isoformat(), "train_cutoff": start.isoformat(),
                "holdout_start": heldout.date.min().isoformat(), "holdout_end": heldout.date.max().isoformat(),
                "n_train": len(train), "n_holdout": len(heldout), "models": {}}
        fitted = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CoverageWarning)
            for name, candidate in (("poisson", GoalModel()), ("baseline", BaselineModel()),
                                    ("dixon_coles", DixonColesModel(max_iter=150))):
                candidate.fit(train, start.isoformat())
                fitted[name] = candidate
                metrics = _metrics(candidate, heldout, 10)
                metrics["ece"] = float(np.mean([
                    sum(b["count"] * abs(b["mean_probability"] - b["observed_frequency"])
                        for b in bins if b["count"]) / len(heldout)
                    for bins in metrics["calibration_bins"].values()
                ]))
                fold["models"][name] = metrics
            weights = np.linspace(0, 1, 21)
            ensembles = []
            for weight in weights:
                candidate = EnsembleModel([fitted["poisson"], fitted["dixon_coles"]], [weight, 1 - weight])
                metrics = _metrics(candidate, heldout, 10)
                metrics["ece"] = float(np.mean([
                    sum(b["count"] * abs(b["mean_probability"] - b["observed_frequency"])
                        for b in bins if b["count"]) / len(heldout)
                    for bins in metrics["calibration_bins"].values()
                ]))
                ensembles.append((metrics["log_loss"], float(weight), metrics))
            _, ensemble_weight, ensemble_metrics = min(ensembles, key=lambda item: item[0])
            ensemble_metrics["poisson_weight"] = ensemble_weight
            fold["models"]["ensemble"] = ensemble_metrics
        report["folds"].append(fold)
    selection = report["folds"][:-1]
    n = sum(f["n_holdout"] for f in selection)
    names = ("poisson", "baseline", "dixon_coles", "ensemble")
    selection_scores = {name: sum(f["n_holdout"] * f["models"][name]["log_loss"] for f in selection) / n
                        for name in names}
    candidate = min(selection_scores, key=selection_scores.get)
    improvement = selection_scores["poisson"] - selection_scores[candidate]
    selected = candidate if (len(selection) >= report["promotion"]["minimum_selection_folds"] and
                             candidate != "poisson" and
                             improvement >= report["promotion"]["minimum_log_loss_improvement"]) else "poisson"
    if selected == "ensemble":
        ensemble_weights = [
            sum(f["n_holdout"] * f["models"]["ensemble"]["poisson_weight"] for f in selection) / n,
        ]
        ensemble_weights.append(1.0 - ensemble_weights[0])
    else:
        ensemble_weights = None
    calibration = report["calibration"]
    if len(selection) >= 2:
        records = []
        try:
            def fitted_selected(train, train_cutoff):
                if selected == "ensemble":
                    return EnsembleModel([GoalModel().fit(train, train_cutoff), DixonColesModel(max_iter=150).fit(train, train_cutoff)], ensemble_weights)
                return _candidate(selected).fit(train, train_cutoff)

            for fold in selection:
                train = frame.loc[frame.date.lt(pd.Timestamp(fold["train_cutoff"]))].copy()
                heldout = ucl.loc[(ucl.date >= pd.Timestamp(fold["holdout_start"])) &
                                  (ucl.date <= pd.Timestamp(fold["holdout_end"]))]
                base = fitted_selected(train, fold["train_cutoff"])
                records.append((*_outcome_records(base, heldout), base))
            train_prob = np.concatenate([item[0] for item in records[:-1]])
            train_outcome = np.concatenate([item[1] for item in records[:-1]])
            valid_prob, valid_outcome, valid_base = records[-1]
            fitted_calibrator = OutcomeCalibrator.fit(train_prob, train_outcome, min_samples=10)
            calibrated_prob = fitted_calibrator.transform_matrix(valid_prob)
            base_metrics = _outcome_metrics(valid_prob, valid_outcome)
            calibrated_metrics = _outcome_metrics(calibrated_prob, valid_outcome)
            promoted = (base_metrics["log_loss"] - calibrated_metrics["log_loss"] >= .005 and
                        calibrated_metrics["brier"] <= base_metrics["brier"])
            assessment_fold = report["folds"][-1]
            assess_train = frame.loc[frame.date.lt(pd.Timestamp(assessment_fold["train_cutoff"]))].copy()
            assess_rows = ucl.loc[(ucl.date >= pd.Timestamp(assessment_fold["holdout_start"])) &
                                  (ucl.date <= pd.Timestamp(assessment_fold["holdout_end"]))]
            assess_base = fitted_selected(assess_train, assessment_fold["train_cutoff"])
            assess_prob, assess_outcome = _outcome_records(assess_base, assess_rows)
            assessment = {"base": _outcome_metrics(assess_prob, assess_outcome),
                          "calibrated": _outcome_metrics(fitted_calibrator.transform_matrix(assess_prob), assess_outcome)}
            calibration = {"status": "ok", "promoted": bool(promoted),
                           "reason": "improved_selection" if promoted else "insufficient_selection_improvement",
                           "minimum_log_loss_improvement": .005,
                           "selection": {"n_train": int(len(train_prob)), "n_validate": int(len(valid_prob)),
                                         "base": base_metrics, "calibrated": calibrated_metrics},
                           "assessment": assessment,
                           "payload": fitted_calibrator.to_dict() if promoted else None}
        except ValueError as exc:
            calibration = {"status": "insufficient_data", "promoted": False,
                           "reason": "insufficient_class_coverage", "payload": None, "detail": str(exc)}
    report.update(status="ok", selected_model=selected, selection_log_loss=selection_scores,
                  assessment=report["folds"][-1]["models"][selected],
                  ensemble_weights=ensemble_weights,
                  calibration=calibration,
                  limitation="Finite UCL holdouts do not establish future reliability; calibration is promoted only on an unseen temporal window.")
    return report


def evaluate_models(history: pd.DataFrame, cutoff: str) -> dict:
    """General comparison plus UCL-specific selection and untouched assessment."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoverageWarning)
        report = evaluate(history, cutoff)
    rolling = evaluate_rolling(history, cutoff)
    report["ucl_rolling"] = rolling
    if rolling["status"] == "ok":
        report["selected_model"] = report["recommended_model"] = rolling["selected_model"]
        report["selection_basis"] = "UCL rolling selection folds; final fold reserved for assessment"
    else:
        report["selection_basis"] = "general holdout; UCL evidence insufficient"
    return report
