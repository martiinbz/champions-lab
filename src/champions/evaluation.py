"""Competition-specific expanding-window validation with a reserved final block."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from champions.models import GoalModel, BaselineModel, CoverageWarning, _completed, _cutoff, _metrics, evaluate
from champions.dixon_coles import DixonColesModel
from champions.ensemble import EnsembleModel


def evaluate_rolling(history: pd.DataFrame, cutoff: str, folds: int = 3) -> dict:
    """Use only UCL holdouts; all competitions may supply pre-block training.

    First folds select the model. The final block is reserved for assessment and
    never changes that selection. Each block refits using strictly earlier data.
    """
    if folds < 2:
        raise ValueError("At least two folds are needed for selection and assessment.")
    report = {"status": "insufficient_data", "folds": [], "selection_folds": folds - 1,
              "assessment_fold": folds, "competition": "UCL",
              "promotion": {"minimum_log_loss_improvement": .005, "minimum_selection_folds": 2}}
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
    report.update(status="ok", selected_model=selected, selection_log_loss=selection_scores,
                  assessment=report["folds"][-1]["models"][selected],
                  ensemble_weights=ensemble_weights,
                  limitation="Finite UCL holdouts do not establish future reliability. ECE is descriptive, not a fitted calibrator.")
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
