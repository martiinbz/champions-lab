"""Explicit updates and immutable, independently reproducible forecast snapshots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd

from champions import __version__


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def as_of(fixtures: pd.DataFrame, cutoff: str, matchday: int | None = None) -> pd.DataFrame:
    """Mask results not yet knowable, including partially played matchdays.

    A conservative 3h completion buffer prevents using a final score at kickoff.
    A date-only historical record is handled upstream as end-of-day.
    """
    frame = fixtures.copy()
    dates = pd.to_datetime(frame.date, utc=True, errors="raise")
    visible = (dates + pd.Timedelta(hours=3) < utc(cutoff))
    if matchday is not None:
        if not 0 <= matchday <= 8:
            raise ValueError("La jornada debe estar entre 0 y 8.")
        visible &= frame.matchday <= matchday
    if "status" in frame:
        visible &= frame.status.eq("finished")
    frame.loc[~visible, ["home_goals", "away_goals"]] = np.nan
    frame["status"] = np.where(frame.home_goals.notna() & frame.away_goals.notna(), "finished", "scheduled")
    return frame


def training_data(history: pd.DataFrame, fixtures: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Merge current European results and deduplicate overlaps across providers."""
    current = fixtures.loc[fixtures.home_goals.notna()].copy()
    current["competition"] = "UCL"
    frame = pd.concat([history.copy(), current], ignore_index=True)
    frame["date"] = pd.to_datetime(frame.date, utc=True, errors="raise")
    frame = frame.loc[(frame.date + pd.Timedelta(hours=3) < utc(cutoff)) &
                      frame.home_goals.notna() & frame.away_goals.notna()].copy()
    frame["_day"] = frame.date.dt.strftime("%Y-%m-%d")
    keys = ["_day", "home", "away"]
    conflicts = frame.groupby(keys)[["home_goals", "away_goals"]].nunique().max(axis=1)
    if (conflicts > 1).any():
        raise ValueError("Fuentes discrepan en un marcador histórico: revisar antes de entrenar.")
    return frame.drop_duplicates(keys, keep="last").drop(columns="_day").sort_values("date").reset_index(drop=True)


def update_data(root: Path) -> dict:
    from champions.data import fetch_current, fetch_history, validate_fixtures
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    raw = root / "data" / "raw" / stamp
    raw.mkdir(parents=True, exist_ok=False)
    fixtures = fetch_current(raw)
    validate_fixtures(fixtures)
    history = fetch_history(raw)
    if history.empty:
        raise ValueError("No hay históricos para entrenar.")
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    generation = processed / "generations" / stamp
    generation.mkdir(parents=True, exist_ok=False)
    # Publish a single pointer only once both inputs and provenance are durable.
    for name, frame in (("fixtures", fixtures), ("history", history)):
        frame.to_csv(generation / f"{name}.csv", index=False)
    report = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "raw_dir": str(raw.relative_to(root)),
              "fixtures": len(fixtures), "history_matches": len(history),
              "latest_history": str(pd.to_datetime(history.date, utc=True).max()),
              "files": {n: sha256(generation / n) for n in ("fixtures.csv", "history.csv")},
              "fixture_attrs": fixtures.attrs, "history_attrs": history.attrs,
              "sources": [json.loads(p.read_text(encoding="utf-8")) for p in raw.rglob("*.meta.json")]}
    write_json(generation / "provenance.json", report)
    pointer = processed / (".current-" + stamp + ".json")
    write_json(pointer, {"generation": stamp})
    os.replace(pointer, processed / "current.json")
    return report


def run(root: Path, cutoff: str, matchday: int | None = None, simulations: int = 10000,
        seed: int = 42, evaluate_model: bool = True, model_kind: str = "auto") -> Path:
    from champions.data import validate_fixtures
    from champions.models import GoalModel
    from champions.evaluation import evaluate_models
    from champions.simulation import simulate
    if simulations < 100 or simulations > 1_000_000:
        raise ValueError("Usa entre 100 y 1.000.000 simulaciones.")
    if utc(cutoff) > pd.Timestamp.now(tz="UTC"):
        raise ValueError("El corte no puede ser futuro: todavía no existen esos datos.")
    inputs = root / "data" / "processed"
    if (inputs / "current.json").exists():
        generation = json.loads((inputs / "current.json").read_text(encoding="utf-8"))["generation"]
        if Path(generation).name != generation or generation.startswith("."):
            raise ValueError("Generación de datos inválida.")
        inputs = inputs / "generations" / generation
    if not (inputs / "fixtures.csv").exists() or not (inputs / "history.csv").exists():
        raise ValueError("Faltan datos. Ejecuta primero: champions update")
    provenance = inputs / "provenance.json"
    source_meta = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {}
    for name, digest in source_meta.get("files", {}).items():
        if Path(name).name != name or sha256(inputs / name) != digest:
            raise ValueError("Los datos descargados han cambiado respecto a su manifiesto.")
    fixtures = pd.read_csv(inputs / "fixtures.csv")
    validate_fixtures(fixtures)
    fixtures = as_of(fixtures, cutoff, matchday)
    history = pd.read_csv(inputs / "history.csv")
    # For explicit historical matchdays, forbid later training fixtures as well.
    if matchday is not None:
        later = fixtures.loc[fixtures.matchday > matchday, "date"]
        if not later.empty and utc(cutoff) > pd.to_datetime(later, utc=True).min():
            raise ValueError("El corte incluye jornadas posteriores a la seleccionada. Usa un corte anterior a su inicio.")
    history = training_data(history, fixtures, cutoff)
    if len(history) < 100:
        raise ValueError("Histórico insuficiente: se requieren al menos 100 partidos reales.")
    teams = sorted(set(fixtures.home) | set(fixtures.away))
    coverage = []
    warnings = []
    for key in ("fixture_attrs", "history_attrs"):
        warnings.extend(source_meta.get(key, {}).get("warnings", []))
        quality = source_meta.get(key, {}).get("quality_report", {})
        for failure in quality.get("source_failures", []):
            warnings.append(f"Fuente no disponible: {failure['source']}: {failure['error']}")
        if quality.get("fallback"):
            warnings.append(quality["fallback"])
    for team in teams:
        rows = history.loc[history.home.eq(team) | history.away.eq(team)]
        coverage.append({"team": team, "matches": len(rows), "latest": str(rows.date.max()) if len(rows) else None})
        if len(rows) < 20:
            warnings.append(f"{team}: solo {len(rows)} partidos históricos; estimación con mayor incertidumbre.")
        if len(rows) and utc(cutoff) - rows.date.max() > pd.Timedelta(days=45):
            warnings.append(f"{team}: último resultado disponible anterior en más de 45 días al corte.")
    # Full source history before the cutoff is retained: no hidden training data.
    evaluation = evaluate_models(history, cutoff) if evaluate_model else {"status": "not_evaluated"}
    requested_model = model_kind
    if model_kind == "auto":
        if not evaluate_model:
            raise ValueError("El modelo auto requiere evaluación; elige --model poisson o baseline al omitirla.")
        model_kind = evaluation.get("selected_model", "baseline")
    model = GoalModel(kind=model_kind).fit(history, cutoff)
    probabilities, diagnostics = simulate(fixtures, model, simulations=simulations, seed=seed)
    warnings.extend(diagnostics.get("warnings", []))
    warnings.extend(getattr(model, "warnings", []))
    for column in ["top8", "positions9_16", "positions17_24", "playoff", "eliminated",
                   "round16", "quarterfinal", "semifinal", "final", "champion"]:
        p = probabilities[column]
        probabilities[column + "_mc_se"] = np.sqrt(p * (1 - p) / simulations)
    played = fixtures.loc[fixtures.status.eq("finished")]
    inferred = int(played.matchday.max()) if not played.empty else 0
    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    target = root / "results" / "snapshots" / run_id
    staging = target.parent / ("." + run_id)
    staging.mkdir(parents=True, exist_ok=False)
    data_target = root / "data" / "snapshots" / run_id
    data_target.mkdir(parents=True, exist_ok=False)
    fixtures.to_csv(data_target / "fixtures.csv", index=False)
    history.to_csv(data_target / "history.csv", index=False)
    pd.DataFrame(coverage).to_csv(data_target / "coverage.csv", index=False)
    write_json(staging / "model.json", model.to_dict())
    probabilities.to_csv(staging / "probabilities.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    metadata = {"run_id": run_id, "season": str(fixtures.season.iloc[0]), "matchday": inferred,
                "requested_matchday": matchday, "cutoff": utc(cutoff).isoformat(), "created_at": created.isoformat(),
                "model": model_kind, "requested_model": requested_model, "version": __version__, "git_commit": commit,
                "simulations": simulations, "seed": seed, "warnings": sorted(set(warnings)),
                "evaluation": evaluation, "coverage": coverage, "diagnostics": diagnostics,
                "known_results": len(played), "history_matches": len(history),
                "latest_training_match": history.date.max().isoformat(),
                "data_snapshot": str(data_target.relative_to(root)),
                "data_hashes": {n: sha256(data_target / n) for n in ("fixtures.csv", "history.csv", "coverage.csv")},
                "output_hashes": {n: sha256(staging / n) for n in ("model.json", "probabilities.csv")},
                "environment": {"python": platform.python_version(), **{n: version(n) for n in ["pandas", "numpy", "scipy"]}},
                "retrospective": utc(cutoff) < created - pd.Timedelta(days=1)}
    if source_meta:
        metadata["source_provenance"] = source_meta
    write_json(staging / "metadata.json", metadata)
    staging.rename(target)
    return target


def reproduce(root: Path, run_id: str) -> dict:
    from champions.models import GoalModel
    from champions.simulation import simulate
    if Path(run_id).name != run_id or run_id.startswith("."):
        raise ValueError("Identificador de ejecución inválido.")
    result = root / "results" / "snapshots" / run_id
    meta = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
    data = (root / meta["data_snapshot"]).resolve()
    if not data.is_relative_to((root / "data" / "snapshots").resolve()):
        raise ValueError("Ruta de snapshot inválida.")
    for name, expected in meta["data_hashes"].items():
        if sha256(data / name) != expected:
            raise ValueError(f"Snapshot alterado: {name}")
    for name, expected in meta["output_hashes"].items():
        if sha256(result / name) != expected:
            raise ValueError(f"Resultado alterado: {name}")
    model = GoalModel.from_dict(json.loads((result / "model.json").read_text(encoding="utf-8")))
    fresh, _ = simulate(pd.read_csv(data / "fixtures.csv"), model, simulations=meta["simulations"], seed=meta["seed"])
    old = pd.read_csv(result / "probabilities.csv")
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), old[fresh.columns].reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    return {"run_id": run_id, "hashes_valid": True, "probabilities_reproduced": True}
