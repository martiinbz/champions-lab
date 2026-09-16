import numpy as np
import pandas as pd
import pytest
from champions.evaluation import evaluate_rolling


def test_final_block_cannot_change_model_selection():
    rng = np.random.default_rng(88)
    rows = []
    for i in range(360):
        h, a = rng.choice(["a", "b", "c", "d"], 2, replace=False)
        rows.append(dict(date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), home=h, away=a,
                         home_goals=int(rng.poisson(2 if h == 'a' else .8)),
                         away_goals=int(rng.poisson(2 if a == 'a' else .8)), competition="UCL"))
    frame = pd.DataFrame(rows)
    report = evaluate_rolling(frame, "2021-01-01")
    assert report["status"] == "ok"
    assert len(report["folds"]) == 3
    frame.loc[270:, ["home_goals", "away_goals"]] = [9, 0]
    perturbed = evaluate_rolling(frame, "2021-01-01")
    assert report["selected_model"] == perturbed["selected_model"]
    assert report["selection_log_loss"] == perturbed["selection_log_loss"]
    assert report["assessment"]["log_loss"] != perturbed["assessment"]["log_loss"]
    for fold in report["folds"]:
        assert fold["train_last_date"] < fold["holdout_start"]
        assert 0 <= fold["models"]["poisson"]["ece"] <= 1


def test_evaluation_reports_insufficient_competition_evidence():
    assert evaluate_rolling(pd.DataFrame(), "2026-09-16")["status"] == "insufficient_data"
    with pytest.raises(ValueError):
        evaluate_rolling(pd.DataFrame(), "2026-09-16", folds=1)
