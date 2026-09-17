"""Read-only, offline dashboard for the Champions snapshot contract.

Probabilities are fractions in [0, 1]. Missing optional fields stay missing.
Run with ``streamlit run app/streamlit_app.py``.
"""
import base64
import hashlib
import html
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
PROBABILITY_ORDER = [
    "champion", "final", "semifinal", "quarterfinal", "round16", "playoff",
    "top8", "positions9_16", "positions17_24", "eliminated",
]
CREST_DIR = Path(__file__).resolve().parent / "assets" / "crests"


def timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def format_integer(value):
    """Format optional integral metadata without letting malformed runs break the UI."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "—"
    if not np.isfinite(number) or not number.is_integer():
        return "—"
    return f"{int(number):,}".replace(",", ".")


def load_snapshots(root):
    """Isolate incomplete/invalid runs; never cache files that may be replaced."""
    snapshots, issues = [], []
    for folder in sorted((root / "results" / "snapshots").glob("*")):
        if not folder.is_dir() or folder.name.startswith('.'):
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
    result = data[["team", *[c for c in PROBABILITY_ORDER if c in data],
                   *[c for c in EXPECTED if c in data]]].copy()
    for column in LABELS:
        if column in result:
            result[column] *= 100
    return result.rename(columns={"team": "Equipo", **LABELS, **EXPECTED})


def expected_ranking(data):
    """Return the full projected league table in display order."""
    if "expected_rank" not in data:
        return pd.DataFrame(columns=["Puesto", "Equipo", "Posición esperada"])
    columns = ["team", "expected_rank"]
    if "expected_points" in data:
        columns.append("expected_points")
    ranking = data[columns].dropna(subset=["expected_rank"]).sort_values(
        ["expected_rank", "team"], kind="stable").reset_index(drop=True)
    ranking.insert(0, "Puesto", np.arange(1, len(ranking) + 1))
    return ranking.rename(columns={"team": "Equipo", **EXPECTED})


def expected_position_history(snapshots):
    """Build one position row per team and unique input-data state."""
    rows, selected = [], {}
    ordered = sorted(snapshots, key=lambda item: (
        timestamp(item[0].get("created_at")).value,
        item[0].get("run_id", ""),
    ))
    for meta, data in ordered:
        if "expected_rank" not in data:
            continue
        hashes = meta.get("data_hashes")
        if isinstance(hashes, dict) and hashes:
            key = ("data", tuple(sorted((str(name), str(digest))
                                        for name, digest in hashes.items())))
        else:
            key = ("legacy", meta.get("cutoff"), meta.get("matchday"))
        selected[key] = (meta, data)
    selected_runs = sorted(selected.values(), key=lambda item: (
        timestamp(item[0].get("cutoff")).value,
        timestamp(item[0].get("created_at")).value,
        item[0].get("run_id", ""),
    ))
    for meta, data in selected_runs:
        for row in data[["team", "expected_rank"]].itertuples(index=False):
            if pd.isna(row.expected_rank):
                continue
            rows.append({
                "Equipo": row.team,
                "Corte": timestamp(meta.get("cutoff")),
                "Posición esperada": float(row.expected_rank),
                "Jornada": int(meta.get("matchday", 0)),
                "Ejecución": meta.get("run_id", ""),
            })
    return pd.DataFrame(rows, columns=[
        "Equipo", "Corte", "Posición esperada", "Jornada", "Ejecución",
    ])


def crest_data_uri(team, crest_dir=CREST_DIR):
    """Return a local crest as a data URI, or a deterministic offline badge."""
    crest_root = Path(crest_dir).resolve()
    mapping_path = crest_root / "team-crests.json"
    if mapping_path.exists():
        try:
            filename = json.loads(mapping_path.read_text(encoding="utf-8")).get(team)
            image_path = (crest_root / filename).resolve() if filename else None
            if image_path and image_path.parent == crest_root and image_path.is_file():
                mime = "image/svg+xml" if image_path.suffix.lower() == ".svg" else "image/png"
                payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
                return f"data:{mime};base64,{payload}"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    initials = "".join(part[0] for part in str(team).replace("/", " ").split()[:3]).upper() or "?"
    hue = int(hashlib.sha256(str(team).encode("utf-8")).hexdigest()[:4], 16) % 360
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
           f'<circle cx="32" cy="32" r="29" fill="hsl({hue} 58% 34%)" stroke="white" stroke-width="3"/>'
           f'<text x="32" y="38" text-anchor="middle" font-family="Arial" font-size="18" '
           f'font-weight="700" fill="white">{initials}</text></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def render_table(data):
    table = probability_table(data)
    table["Escudo"] = table["Equipo"].map(crest_data_uri)
    st.dataframe(table, hide_index=True, width="stretch", height=min(650, 42 + 35 * len(table)),
                 column_config={
                     "Equipo": st.column_config.TextColumn("Equipo", pinned=True, width="medium"),
                     "Escudo": st.column_config.ImageColumn("", width="small"),
                     **{name: st.column_config.NumberColumn(name, format="%.1f%%")
                        for name in LABELS.values()},
                     **{name: st.column_config.NumberColumn(name, format="%.1f")
                        for name in EXPECTED.values()},
                 })


def render_expected_ranking(data):
    ranking = expected_ranking(data)
    if ranking.empty:
        st.info("La posición esperada no está disponible en esta ejecución.")
        return
    if len(ranking) != 36:
        st.warning(f"Clasificación incompleta: este snapshot contiene {len(ranking)} de 36 equipos.")
        return
    groups = [ranking.iloc[indexes] for indexes in np.array_split(np.arange(len(ranking)), 3)]
    for column, group in zip(st.columns(3), groups):
        with column:
            for _, row in group.iterrows():
                crest = crest_data_uri(row["Equipo"])
                points = row.get("Puntos esperados")
                points_html = (
                    f'<span class="rank-points">{points:.1f} pts</span>'
                    if pd.notna(points) else ""
                )
                st.markdown(
                    '<div class="rank-row">'
                    f'<span class="rank-number">{int(row["Puesto"])}</span>'
                    f'<img src="{crest}" alt="" />'
                    f'<span class="rank-team">{html.escape(row["Equipo"])}</span>'
                    f'{points_html}'
                    '</div>',
                    unsafe_allow_html=True,
                )


def freshness(meta):
    cutoff = timestamp(meta.get("cutoff"))
    created = timestamp(meta.get("created_at"))
    def fmt(value):
        return value.strftime('%d/%m/%Y %H:%M UTC') if not pd.isna(value) else 'No disponible'
    st.caption(f"Corte: {fmt(cutoff)} · Creado: {fmt(created)}")
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


def main():
    st.set_page_config(page_title="Champions · Probabilidades", page_icon="⚽", layout="wide")
    st.markdown("""
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
      h1, h2, h3 {letter-spacing: -0.025em;}
      [data-testid="stMetric"] {background: linear-gradient(135deg,#111827,#182235); border:1px solid #26344c; padding:1rem; border-radius:16px;}
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:#f8fafc;}
      .rank-row {display:flex;align-items:center;gap:.65rem;min-height:48px;padding:.45rem .65rem;margin:.28rem 0;border:1px solid rgba(128,128,128,.22);border-radius:12px;background:rgba(128,128,128,.055)}
      .rank-row img {width:31px;height:31px;object-fit:contain;}
      .rank-number {width:1.7rem;font-weight:800;color:#8b9ab7;text-align:right;}
      .rank-team {font-weight:650;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
      .rank-points {font-size:.78rem;color:#8b9ab7;white-space:nowrap;}
    </style>
    """, unsafe_allow_html=True)
    st.title("Champions 2026/27")
    st.caption("Probabilidades del modelo y clasificación esperada · Champions masculina")
    root = Path(os.environ.get("CHAMPIONS_ROOT") or Path(__file__).resolve().parents[1]).expanduser().resolve()
    st.sidebar.header("Ejecución")
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
    run = st.sidebar.selectbox("Ejecución", list(runs), key="run", format_func=lambda key:
                               f"{runs[key][0].get('model', 'Modelo')} · {runs[key][0].get('simulations', '?')} sims · {key}")
    meta, data = runs[run]
    selected = st.sidebar.multiselect("Filtrar tabla", sorted(data.team), key="teams",
                                      help="La clasificación y la evolución siempre muestran los 36 equipos.")
    filtered = data[data.team.isin(selected)] if selected else data
    if "champion" in filtered:
        filtered = filtered.sort_values("champion", ascending=False)
    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Temporada", season)
    card2.metric("Jornada", day)
    card3.metric("Simulaciones", format_integer(meta.get("simulations")))
    card4.metric("Partidos jugados", format_integer(meta.get("known_results")))
    freshness(meta)
    warnings = meta.get("warnings") or []
    if warnings:
        with st.expander("Cobertura, supuestos y limitaciones del modelo", expanded=False):
            for warning in warnings if isinstance(warnings, list) else [warnings]:
                st.warning(str(warning))
    st.subheader("Probabilidades por equipo")
    st.caption("De campeón a eliminación: los hitos están ordenados de más difícil a más accesible. Probabilidades en %.")
    render_table(filtered)
    missing = [label for field, label in {**LABELS, **EXPECTED}.items() if field not in data]
    if missing:
        st.caption("Columnas no disponibles: " + ", ".join(missing))

    st.divider()
    st.subheader("Clasificación esperada")
    st.caption("Orden medio proyectado al terminar la fase liga para los 36 equipos.")
    render_expected_ranking(data)

    st.divider()
    st.subheader("Evolución de la posición esperada")
    history = expected_position_history(season_runs).dropna(subset=["Corte"])
    if history.empty:
        st.info("Se necesitan al menos dos snapshots con posición esperada para dibujar la evolución.")
    else:
        fig = px.line(
            history, x="Corte", y="Posición esperada", color="Equipo", markers=True,
            hover_data={"Jornada": True, "Ejecución": False, "Posición esperada": ":.1f"},
            template="plotly_white",
        )
        fig.update_traces(line={"width": 1.55}, marker={"size": 5})
        fig.update_layout(
            height=680, showlegend=False, hovermode="closest", margin={"l": 35, "r": 20, "t": 15, "b": 35},
            yaxis={"autorange": "reversed", "dtick": 2, "range": [36.5, .5], "title": "Posición esperada"},
            xaxis={"title": None},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Cada línea es un equipo. Pasa el cursor por un punto para identificarlo; 1 es la mejor posición.")

    with st.expander("Metodología y detalles técnicos", expanded=False):
        st.markdown("**Incertidumbre Monte Carlo**: más simulaciones reducen el error numérico, pero no corrigen posibles sesgos del modelo.")
        st.markdown("La **calibración del modelo** usa evaluación fuera de muestra para comprobar si las probabilidades se parecen a las frecuencias observadas.")
        if meta.get("evaluation"):
            st.json(meta["evaluation"])
        else:
            st.info("Sin evaluación registrada: no se puede determinar la calibración del modelo.")
        st.json({field: meta.get(field) for field in (
            "run_id", "season", "matchday", "cutoff", "created_at", "model",
            "simulations", "seed", "coverage")})
        st.caption(f"Origen local: {root / 'results' / 'snapshots' / run}")
    st.caption("Estimaciones estadísticas condicionadas al modelo y a los datos disponibles.")


if __name__ == "__main__":
    main()
