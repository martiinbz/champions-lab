"""Read-only, offline dashboard for the Champions snapshot contract.

Probabilities are fractions in [0, 1]. Missing optional fields stay missing.
Run with ``streamlit run app/streamlit_app.py``.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


LABELS = {
    "top8": "Top 8", "positions9_16": "Puestos 9–16",
    "positions17_24": "Puestos 17–24", "playoff": "Playoff",
    "eliminated": "Eliminado", "round16": "Octavos",
    "quarterfinal": "Cuartos", "semifinal": "Semifinal",
    "final": "Final", "champion": "Campeón",
}
EXPECTED = {"expected_points": "Puntos esperados", "expected_rank": "Posición esperada"}


def timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def load_snapshots(root):
    """Isolate incomplete/invalid runs; never cache files that may be replaced."""
    snapshots, issues = [], []
    for folder in sorted((root / "results" / "snapshots").glob("*")):
        if not folder.is_dir():
            continue
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
            if not isinstance(meta, dict):
                raise ValueError("los metadatos deben ser un objeto")
            for field in ("run_id", "season", "matchday"):
                if meta.get(field) is None or str(meta[field]).strip() == "":
                    raise ValueError(f"falta {field}")
            if str(meta["run_id"]) != folder.name:
                raise ValueError("run_id no coincide con la carpeta")
            meta["season"] = str(meta["season"])
            day = float(meta["matchday"])
            if not np.isfinite(day) or day < 0 or not day.is_integer():
                raise ValueError("jornada inválida")
            meta["matchday"] = int(day)
            data = pd.read_csv(folder / "probabilities.csv", dtype={"team": str})
            if "team" not in data or data.empty:
                raise ValueError("faltan equipos")
            if data.team.isna().any() or data.team.str.strip().eq("").any():
                raise ValueError("hay equipos sin nombre")
            data["team"] = data.team.str.strip()
            if data.team.duplicated().any():
                raise ValueError("hay equipos duplicados")
            for field in [*LABELS, *EXPECTED]:
                if field not in data:
                    continue
                original = data[field]
                numbers = pd.to_numeric(original, errors="coerce")
                if (original.notna() & numbers.isna()).any():
                    raise ValueError(f"{field}: valor no numérico")
                present = numbers.dropna()
                if not np.isfinite(present).all():
                    raise ValueError(f"{field}: valor no finito")
                if field in LABELS and not present.between(0, 1).all():
                    raise ValueError(f"{field}: probabilidad fuera de [0, 1]")
                data[field] = numbers
            snapshots.append((meta, data))
        except (OSError, ValueError, TypeError, OverflowError, pd.errors.ParserError) as exc:
            issues.append(f"Snapshot {folder.name} omitido: {exc}")
    return sorted(snapshots, key=lambda item: (
        timestamp(item[0].get("created_at")).value,
        item[0]["run_id"],
    ), reverse=True), issues


def probability_table(data):
    result = data[["team", *[c for c in LABELS if c in data],
                   *[c for c in EXPECTED if c in data]]].copy()
    for column in LABELS:
        if column in result:
            result[column] *= 100
    return result.rename(columns={"team": "Equipo", **LABELS, **EXPECTED})


def render_table(data):
    st.dataframe(probability_table(data), hide_index=True, width="stretch",
                 column_config={
                     **{name: st.column_config.NumberColumn(name, format="%.1f%%")
                        for name in LABELS.values()},
                     **{name: st.column_config.NumberColumn(name, format="%.1f")
                        for name in EXPECTED.values()},
                 })


def freshness(meta):
    cutoff = timestamp(meta.get("cutoff"))
    created = timestamp(meta.get("created_at"))
    st.caption(f"Corte de los datos: {meta.get('cutoff') or 'No disponible'} · "
               f"Ejecución creada: {meta.get('created_at') or 'No disponible'}")
    if pd.isna(cutoff):
        st.warning("Antigüedad de los datos desconocida: falta una fecha de corte válida.")
    else:
        days = (pd.Timestamp.now(tz="UTC") - cutoff).total_seconds() / 86400
        if days < 0:
            st.warning("El corte de datos está en el futuro; revisa los metadatos.")
        else:
            st.caption(f"Antigüedad del corte: {days:.1f} días. "
                       "Los resultados no se actualizan automáticamente con nuevos partidos.")
    if pd.isna(created):
        st.caption("Fecha de creación no disponible o inválida.")


def detail(data, meta):
    st.subheader("Detalle del equipo")
    if data.empty:
        st.info("Selecciona algún equipo para ver su detalle.")
        return
    team = st.selectbox("Equipo en detalle", data.team.tolist(), key="detail")
    row = data.set_index("team").loc[team]
    records = [{"Hito": label, "Probabilidad (%)": row[field] * 100}
               for field, label in LABELS.items() if field in row and pd.notna(row[field])]
    if not records:
        st.info("No hay probabilidades disponibles para este equipo.")
        return
    values = pd.DataFrame(records)
    # Wilson intervals retain nonzero width at p=0 and p=1.
    try:
        n = float(meta.get("simulations"))
        valid_n = np.isfinite(n) and n > 0 and n.is_integer()
    except (TypeError, ValueError, OverflowError):
        valid_n = False
    if valid_n:
        p = values["Probabilidad (%)"] / 100
        z = 1.959963984540054
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        values["MC 95% inferior (%)"] = (center - half).clip(0, 1) * 100
        values["MC 95% superior (%)"] = (center + half).clip(0, 1) * 100
    else:
        st.info("Intervalos Monte Carlo no disponibles: número de simulaciones desconocido.")
    st.dataframe(values, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.2f")
                                for c in values.columns if c != "Hito"})


def main():
    st.set_page_config(page_title="Champions · Probabilidades", page_icon="⚽", layout="wide")
    st.title("Champions · Probabilidades")
    st.caption("Una mirada a la competición, ejecución a ejecución. Resultados locales del modelo.")
    root = Path(os.environ.get("CHAMPIONS_ROOT") or Path(__file__).resolve().parents[1]).expanduser().resolve()
    st.sidebar.header("Explorar resultados")
    st.sidebar.button("Actualizar lista", key="refresh")
    snapshots, issues = load_snapshots(root)
    for issue in issues:
        st.warning(issue)
    if not snapshots:
        st.info("No hay snapshots válidos. Genera resultados con el pipeline para comenzar.")
        st.markdown("Desde la raíz del proyecto, descarga los datos y genera una ejecución. "
                    "La descarga se realiza únicamente al ejecutar el comando update:")
        st.code("champions --help\nchampions update\nchampions simulate --simulations 10000 --seed 42\n"
                "python -m streamlit run app/streamlit_app.py", language="bash")
        st.caption("Si usas CHAMPIONS_ROOT, pasa también esa carpeta con champions --root RUTA "
                   "antes de update o simulate.")
        st.caption(f"Ruta esperada: {root / 'results' / 'snapshots'} / <run_id> / "
                   "metadata.json y probabilities.csv. También puedes configurar CHAMPIONS_ROOT.")
        return

    season = st.sidebar.selectbox("Temporada", sorted({m["season"] for m, _ in snapshots}, reverse=True), key="season")
    season_runs = [(m, d) for m, d in snapshots if m["season"] == season]
    day = st.sidebar.selectbox("Jornada", sorted({m["matchday"] for m, _ in season_runs}, reverse=True), key="matchday")
    runs = {m["run_id"]: (m, d) for m, d in season_runs if m["matchday"] == day}
    run = st.sidebar.selectbox("Ejecución", list(runs), key="run")
    meta, data = runs[run]
    selected = st.sidebar.multiselect("Filtrar equipos", sorted(data.team), key="teams",
                                      help="Sin selección se muestran todos los equipos.")
    filtered = data[data.team.isin(selected)] if selected else data
    freshness(meta)
    warnings = meta.get("warnings") or []
    for warning in warnings if isinstance(warnings, list) else [warnings]:
        st.warning(str(warning))
    st.subheader("Probabilidades por equipo")
    st.caption("Probabilidades en %. Haz clic en las cabeceras para ordenar. — indica un dato ausente.")
    render_table(filtered)
    missing = [label for field, label in {**LABELS, **EXPECTED}.items() if field not in data]
    if missing:
        st.caption("Columnas no disponibles: " + ", ".join(missing))

    history_tab, detail_tab, comparison_tab, metadata_tab = st.tabs(
        ["Evolución", "Detalle e incertidumbre", "Comparar ejecuciones", "Metadatos"])
    with history_tab:
        st.subheader("Evolución histórica")
        metrics = [field for field in LABELS if any(field in d for _, d in season_runs)]
        if metrics:
            metric = st.selectbox("Probabilidad a seguir", metrics, format_func=LABELS.get, key="history_metric")
            rows = []
            for m, d in season_runs:
                if metric not in d:
                    continue
                for _, row in d.iterrows():
                    if selected and row.team not in selected:
                        continue
                    rows.append({"Equipo": row.team, "Corte": timestamp(m.get("cutoff")),
                                 "Probabilidad (%)": row[metric] * 100,
                                 "Jornada": m["matchday"], "Ejecución": m["run_id"],
                                 "Creación": timestamp(m.get("created_at"))})
            history = pd.DataFrame(rows)
            if not history.empty:
                missing_dates = history.Corte.isna().any()
                history = history.dropna(subset=["Corte"]).sort_values(["Corte", "Creación", "Ejecución"])
                if missing_dates:
                    st.caption("Se omiten del gráfico los cortes sin fecha válida.")
                if not history.empty:
                    fig = px.line(history, x="Corte", y="Probabilidad (%)", color="Equipo", markers=True,
                                  hover_data=["Jornada", "Ejecución"], template="plotly_white")
                    fig.update_layout(yaxis_range=[0, 100], legend_title_text="Equipo", hovermode="closest")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No hay fechas de corte válidas para dibujar la evolución.")
            st.caption("Historial de toda la temporada seleccionada. Cada punto identifica una ejecución; "
                       "los cambios también pueden reflejar distintas versiones del modelo.")
        else:
            st.info("No hay probabilidades disponibles para mostrar la evolución.")
    with detail_tab:
        st.markdown("**Incertidumbre Monte Carlo**: los intervalos Wilson del 95% estiman el error "
                    "numérico por un número finito de simulaciones independientes, con el modelo fijo. "
                    "Más simulaciones reducen este error; no garantizan mejores predicciones.")
        st.markdown("**Calibración del modelo**: mide si las probabilidades concuerdan con frecuencias "
                    "observadas en evaluación fuera de muestra. Los intervalos Monte Carlo no incluyen "
                    "sesgos del modelo, cambios de equipos ni datos incompletos, y no acreditan calibración.")
        detail(filtered, meta)
        st.markdown("**Evaluación del modelo registrada**")
        if meta.get("evaluation"):
            st.json(meta["evaluation"])
        else:
            st.info("Sin evaluación registrada: no se puede determinar la calibración del modelo.")
    with comparison_tab:
        others = {m["run_id"]: (m, d) for m, d in season_runs if m["run_id"] != run}
        if not others:
            st.info("Se necesitan dos ejecuciones de la misma temporada para comparar.")
        else:
            baseline = st.selectbox("Ejecución de referencia", list(others), key="baseline")
            base_meta, base_data = others[baseline]
            common = [c for c in LABELS if c in data and c in base_data]
            st.caption(f"Actual: {run} · corte {meta.get('cutoff', 'No disponible')} | "
                       f"Referencia: {baseline} · corte {base_meta.get('cutoff', 'No disponible')}. "
                       "Cambio = actual − referencia, en puntos porcentuales (pp).")
            if meta.get("model") != base_meta.get("model"):
                st.warning("Las ejecuciones utilizan modelos distintos; el cambio no se debe solo a nuevos resultados.")
            if common:
                metric = st.selectbox("Probabilidad a comparar", common, format_func=LABELS.get, key="compare_metric",
                                      index=common.index("champion") if "champion" in common else 0)
                comparison = data.set_index("team")[[metric]].rename(columns={metric: "Actual (%)"}).join(
                    base_data.set_index("team")[[metric]].rename(columns={metric: "Referencia (%)"}), how="outer") * 100
                if selected:
                    comparison = comparison.loc[comparison.index.isin(selected)]
                comparison["Cambio (pp)"] = comparison["Actual (%)"] - comparison["Referencia (%)"]
                st.dataframe(comparison.rename_axis("Equipo").reset_index(), hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in comparison})
                st.caption("Un equipo ausente en una ejecución mantiene su valor vacío, sin asumir un 0%.")
            else:
                st.info("No hay columnas de probabilidad comunes entre estas ejecuciones.")
    with metadata_tab:
        st.subheader("Ficha de la ejecución")
        st.caption("Modelo, simulaciones, semilla, cobertura y evaluación tal como los guardó el pipeline.")
        st.json({field: meta.get(field) for field in (
            "run_id", "season", "matchday", "cutoff", "created_at", "model",
            "simulations", "seed", "warnings", "evaluation", "coverage")})
        st.caption(f"Origen local: {root / 'results' / 'snapshots' / run}")
    st.caption("Estimaciones estadísticas condicionadas al modelo y a los datos disponibles.")


if __name__ == "__main__":
    main()
