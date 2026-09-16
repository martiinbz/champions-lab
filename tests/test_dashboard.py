"""Dashboard integration tests using only temporary, local snapshots."""
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def snapshot(root, run="run-1", season="2026/27", day=1, probability=.2,
             optional=True):
    folder = root / "results" / "snapshots" / run
    folder.mkdir(parents=True)
    metadata = dict(run_id=run, season=season, matchday=day,
                    cutoff=f"2026-09-{day:02d}T18:00:00Z",
                    created_at=f"2026-09-{day:02d}T20:00:00Z",
                    model="poisson-v1", simulations=10000, seed=42,
                    warnings=["Cobertura parcial"], evaluation={"brier": .18},
                    coverage={"matches": 18})
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = [{"team": "Real Madrid", "champion": probability},
            {"team": "Barcelona", "champion": .1}]
    if optional:
        for row in rows:
            row.update(top8=.6, positions9_16=.2, positions17_24=.1,
                       playoff=.3, eliminated=.1, round16=.75,
                       quarterfinal=.5, semifinal=.4, final=.3,
                       expected_points=17.5, expected_rank=6.2)
    pd.DataFrame(rows).to_csv(folder / "probabilities.csv", index=False)
    return folder


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAMPIONS_ROOT", str(tmp_path))
    return tmp_path


def start():
    assert APP.exists(), "Dashboard entry point must exist"
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    return app


def test_empty_state_has_commands_without_percentages(root):
    app = start()
    assert any("No hay snapshots" in item.value for item in app.info)
    assert "champions --help" in app.code[0].value
    assert not app.dataframe
    assert not app.metric


def test_contract_labels_freshness_metadata_and_uncertainty(root):
    snapshot(root)
    app = start()
    table = app.dataframe[0].value
    assert table.loc[0, "Equipo"] == "Real Madrid"
    assert table.loc[0, "Campeón"] == pytest.approx(20)
    assert "Puestos 17–24" in table
    text = " ".join(x.value for x in app.caption) + " ".join(x.value for x in app.markdown)
    assert "01/09/2026" in text
    assert "Monte Carlo" in text and "calibración" in text
    assert any("Cobertura parcial" in x.value for x in app.warning)
    assert app.json


def test_selectors_filter_history_and_comparison(root):
    snapshot(root)
    snapshot(root, "run-2", day=2, probability=.3)
    snapshot(root, "run-3", season="2025/26", day=8)
    app = start()
    assert app.selectbox(key="season").value == "2026/27"
    assert app.selectbox(key="run").value == "run-2"
    app.multiselect(key="teams").set_value(["Real Madrid"]).run()
    assert len(app.dataframe[0].value) == 1
    assert app.get("plotly_chart")
    comparison = app.dataframe[-1].value
    assert comparison.loc[0, "Cambio (pp)"] == pytest.approx(10)
    app.selectbox(key="matchday").set_value(1).run()
    assert app.selectbox(key="run").value == "run-1"
    app.selectbox(key="season").set_value("2025/26").run()
    assert not app.exception
    assert app.selectbox(key="run").value == "run-3"


def test_optional_columns_are_missing_not_zero(root):
    snapshot(root, optional=False)
    app = start()
    assert "Top 8" not in app.dataframe[0].value
    assert "Campeón" in app.dataframe[0].value
    assert any("no disponibles" in x.value for x in app.caption)


def test_invalid_snapshot_is_skipped_with_warning(root):
    snapshot(root)
    bad = root / "results" / "snapshots" / "broken"
    bad.mkdir()
    (bad / "metadata.json").write_text("{", encoding="utf-8")
    app = start()
    assert any("broken" in item.value for item in app.warning)
    assert app.dataframe


@pytest.mark.parametrize("value", [1.2, -0.1, "oops"])
def test_invalid_probabilities_never_displayed(root, value):
    folder = snapshot(root)
    data = pd.read_csv(folder / "probabilities.csv")
    data["champion"] = value
    data.to_csv(folder / "probabilities.csv", index=False)
    app = start()
    assert not app.dataframe
    assert app.warning


def test_reload_reads_new_snapshots(root):
    snapshot(root)
    app = start()
    snapshot(root, "run-2", day=2)
    app.button(key="refresh").click().run()
    assert 2 in app.selectbox(key="matchday").options or "2" in app.selectbox(key="matchday").options


def test_script_root_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAMPIONS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    # Execute a copy under a temporary project, so real project data is untouched.
    project = tmp_path / "project"
    (project / "app").mkdir(parents=True)
    entry = project / "app" / "streamlit_app.py"
    entry.write_text(APP.read_text(encoding="utf-8"), encoding="utf-8")
    snapshot(project)
    app = AppTest.from_file(str(entry)).run(timeout=20)
    assert not app.exception
    assert app.dataframe[0].value.loc[0, "Campeón"] == pytest.approx(20)


def test_missing_optional_metadata_is_explicit(root):
    folder = snapshot(root)
    (folder / "metadata.json").write_text(json.dumps({
        "run_id": "run-1", "season": "2026/27", "matchday": 1,
    }), encoding="utf-8")
    app = start()
    assert any("Antigüedad" in x.value for x in app.warning)
    assert any("Sin evaluación" in x.value for x in app.info)
    assert any("Intervalos Monte Carlo no disponibles" in x.value for x in app.info)


def test_multiple_runs_same_day_and_absent_team_comparison(root):
    snapshot(root)
    folder = snapshot(root, "run-2", probability=.3)
    data = pd.read_csv(folder / "probabilities.csv")
    data[data.team == "Barcelona"].to_csv(folder / "probabilities.csv", index=False)
    app = start()
    assert app.selectbox(key="run").value == "run-2"
    comparison = app.dataframe[-1].value.set_index("Equipo")
    assert pd.isna(comparison.loc["Real Madrid", "Actual (%)"])
    assert pd.isna(comparison.loc["Real Madrid", "Cambio (pp)"])
    app.selectbox(key="run").set_value("run-1").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 2


def test_offline_and_wilson_interval_at_zero(root, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Dashboard attempted a network connection")

    snapshot(root, probability=0)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    app = start()
    app.selectbox(key="detail").select("Real Madrid").run()
    detail = app.dataframe[1].value.set_index("Hito")
    assert detail.loc["Campeón", "Probabilidad (%)"] == 0
    assert detail.loc["Campeón", "MC 95% superior (%)"] > 0
