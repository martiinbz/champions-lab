"""Static HTML dashboard: bundles the saved snapshots into one self-contained page.

``champions export-web`` reads ``results/snapshots`` and writes ``web/index.html`` from
``web/template.html``. The page needs no server, build step or internet connection.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

PLACEHOLDER = "/*__CHAMPIONS_DATA__*/null"
PROBABILITY_FIELDS = ["top8", "positions9_16", "positions17_24", "playoff", "eliminated",
                      "round16", "quarterfinal", "semifinal", "final", "champion"]


def _clean(value):
    """JSON-safe scalar (NaN/inf become null)."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _crests(root: Path) -> dict[str, str]:
    folder = root / "app" / "assets" / "crests"
    mapping_path = folder / "team-crests.json"
    result = {}
    if not mapping_path.exists():
        return result
    for team, filename in json.loads(mapping_path.read_text(encoding="utf-8")).items():
        path = (folder / filename).resolve()
        if path.parent == folder.resolve() and path.is_file():
            mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
            result[team] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return result


def load_snapshots(root: Path) -> tuple[list[tuple[dict, pd.DataFrame]], list[str]]:
    """Valid snapshots, oldest first; unreadable ones are reported, never fatal."""
    found, issues = [], []
    for folder in sorted((root / "results" / "snapshots").glob("*")):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
            data = pd.read_csv(folder / "probabilities.csv", dtype={"team": str})
            if data.empty or "team" not in data or str(meta["run_id"]) != folder.name:
                raise ValueError("incomplete snapshot")
            found.append((meta, data))
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
            issues.append(f"Snapshot {folder.name} omitido: {exc}")
    found.sort(key=lambda item: (str(item[0].get("created_at")), item[0]["run_id"]))
    return found, issues


def _evolution(snapshots) -> list[dict]:
    """One point per distinct input-data state, mirroring the Streamlit history logic."""
    selected = {}
    for meta, data in snapshots:
        if "expected_rank" not in data:
            continue
        hashes = meta.get("data_hashes")
        key = json.dumps(sorted((hashes or {}).items())) if hashes else f"{meta.get('cutoff')}|{meta.get('matchday')}"
        selected[key] = (meta, data)
    points = []
    for meta, data in sorted(selected.values(), key=lambda m: (str(m[0].get("cutoff")), str(m[0].get("created_at")))):
        points.append({"cutoff": meta.get("cutoff"), "matchday": meta.get("matchday"),
                       "ranks": {r.team: _clean(float(r.expected_rank)) for r in data.itertuples()
                                 if pd.notna(r.expected_rank)}})
    return points


def _matches(root: Path, meta: dict, run: Path) -> list[dict]:
    """League fixtures with the saved model's 1X2 probabilities for unplayed games."""
    from champions.models import GoalModel, outcome_probabilities
    data_dir = root / meta.get("data_snapshot", "")
    fixtures = pd.read_csv(data_dir / "fixtures.csv")
    try:
        model = GoalModel.from_dict(json.loads((run / "model.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        model = None
    rows = []
    for fx in fixtures.sort_values(["matchday", "date", "home"]).itertuples():
        played = pd.notna(fx.home_goals) and pd.notna(fx.away_goals)
        entry = {"matchday": int(fx.matchday), "date": fx.date, "home": fx.home, "away": fx.away,
                 "hg": int(fx.home_goals) if played else None, "ag": int(fx.away_goals) if played else None,
                 "p": None, "xg": None}
        if model is not None and not played:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lam = model.predict_goals(fx.home, fx.away)
                p = outcome_probabilities(*lam)
            entry["p"] = [p["home"], p["draw"], p["away"]]
            entry["xg"] = [float(lam[0]), float(lam[1])]
        rows.append(entry)
    return rows


def _model_summary(meta: dict) -> dict:
    evaluation = meta.get("evaluation") or {}
    rolling = evaluation.get("ucl_rolling") or {}
    calibration = rolling.get("calibration") or {}
    summary = {"selected_model": meta.get("model"), "requested_model": meta.get("requested_model"),
               "selected_candidate": rolling.get("selected_model") or meta.get("model"),
               "basis": evaluation.get("selection_basis"), "status": rolling.get("status"),
               "calibrated": bool(calibration.get("promoted"))}
    if rolling.get("status") == "ok":
        scores = rolling.get("selection_log_loss", {})
        summary["baseline_log_loss"] = scores.get("baseline")
        summary["selected_log_loss"] = scores.get(rolling.get("selected_model"))
        assessment = rolling.get("assessment") or {}
        summary["assessment"] = {k: assessment.get(k) for k in ("log_loss", "brier", "ece")}
        baseline_fold = (rolling.get("folds") or [{}])[-1].get("models", {}).get("baseline", {})
        summary["assessment_baseline"] = {k: baseline_fold.get(k) for k in ("log_loss", "brier", "ece")}
        summary["ranking"] = sorted(scores.items(), key=lambda kv: kv[1])[:6]
    return summary


def build_payload(root: Path) -> dict:
    root = Path(root)
    snapshots, issues = load_snapshots(root)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "issues": issues, "empty": not snapshots}
    if not snapshots:
        return payload
    season = max(str(m.get("season")) for m, _ in snapshots)
    season_runs = [(m, d) for m, d in snapshots if str(m.get("season")) == season]
    # Headline = the forecast with the most recent data cutoff (not the last one computed,
    # which may be a retrospective reconstruction).
    meta, data = max(season_runs, key=lambda m: (pd.to_datetime(m[0].get("cutoff"), utc=True, errors="coerce"), str(m[0].get("created_at"))))
    crests = _crests(root)
    teams = []
    for row in data.sort_values("champion", ascending=False).to_dict("records") if "champion" in data else data.to_dict("records"):
        teams.append({"team": row["team"], "crest": crests.get(row["team"]),
                      **{k: _clean(row.get(k)) for k in (*PROBABILITY_FIELDS, "expected_points", "expected_rank")}})
    payload.update(
        season=season, teams=teams, evolution=_evolution(season_runs),
        run={"run_id": meta["run_id"], "matchday": meta.get("matchday"), "cutoff": meta.get("cutoff"),
             "created_at": meta.get("created_at"), "simulations": meta.get("simulations"),
             "known_results": meta.get("known_results"), "history_matches": meta.get("history_matches"),
             "latest_training_match": meta.get("latest_training_match"),
             "warnings": (meta.get("warnings") or [])[:12]},
        model=_model_summary(meta))
    try:
        payload["matches"] = _matches(root, meta, root / "results" / "snapshots" / meta["run_id"])
    except (OSError, ValueError, KeyError) as exc:
        payload["matches"] = []
        payload["issues"].append(f"Partidos no disponibles: {exc}")
    return payload


def export(root: Path, out: Path | None = None) -> Path:
    root = Path(root)
    template = (root / "web" / "template.html").read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError("web/template.html no contiene el marcador de datos.")
    blob = json.dumps(build_payload(root), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # "</" would let team or warning text close the script element early.
    blob = blob.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    target = Path(out) if out else root / "web" / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace(PLACEHOLDER, blob), encoding="utf-8")
    return target
