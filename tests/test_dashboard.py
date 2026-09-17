"""Dashboard integration tests using only temporary, local snapshots."""
import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def dashboard_module():
    spec = importlib.util.spec_from_file_location("champions_dashboard", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_probability_columns_follow_tournament_from_hardest_to_easiest():
    dashboard = dashboard_module()
    row = {"team": "Barcelona", **{name: .1 for name in dashboard.LABELS},
           "expected_points": 18.2, "expected_rank": 3.4}
    table = dashboard.probability_table(pd.DataFrame([row]))
    assert table.columns.tolist() == [
        "Equipo", "Campeón", "Final", "Semifinal", "Cuartos", "Octavos",
        "Playoff", "Top 8", "Puestos 9–16", "Puestos 17–24", "Eliminado",
        "Puntos esperados", "Posición esperada",
    ]


def test_expected_ranking_orders_every_team_and_numbers_the_places():
    dashboard = dashboard_module()
    data = pd.DataFrame({
        "team": ["Tercero", "Primero", "Segundo"],
        "expected_rank": [3.1, 1.2, 2.4],
        "expected_points": [10, 14, 12],
    })
    ranking = dashboard.expected_ranking(data)
    assert ranking["Equipo"].tolist() == ["Primero", "Segundo", "Tercero"]
    assert ranking["Puesto"].tolist() == [1, 2, 3]


def test_expected_position_history_contains_all_teams_and_snapshots():
    dashboard = dashboard_module()
    runs = [
        ({"run_id": "a", "matchday": 0, "cutoff": "2026-09-01T00:00:00Z",
          "created_at": "2026-09-01T01:00:00Z"},
         pd.DataFrame({"team": ["A", "B"], "expected_rank": [1.5, 2.5]})),
        ({"run_id": "b", "matchday": 1, "cutoff": "2026-09-10T00:00:00Z",
          "created_at": "2026-09-10T01:00:00Z"},
         pd.DataFrame({"team": ["A", "B"], "expected_rank": [2.1, 1.9]})),
    ]
    history = dashboard.expected_position_history(runs)
    assert set(history["Equipo"]) == {"A", "B"}
    assert len(history) == 4
    assert history["Posición esperada"].tolist() == [1.5, 2.5, 2.1, 1.9]


def test_expected_position_history_collapses_reruns_of_same_data_state():
    dashboard = dashboard_module()
    runs = [
        ({"run_id": "old", "matchday": 1, "cutoff": "2026-09-10T00:00:00Z",
          "created_at": "2026-09-10T01:00:00Z", "data_hashes": {"fixtures.csv": "same"}},
         pd.DataFrame({"team": ["A"], "expected_rank": [2.1]})),
        ({"run_id": "new", "matchday": 1, "cutoff": "2026-09-09T00:00:00Z",
          "created_at": "2026-09-11T01:00:00Z", "data_hashes": {"fixtures.csv": "same"}},
         pd.DataFrame({"team": ["A"], "expected_rank": [1.9]})),
    ]
    history = dashboard.expected_position_history(runs)
    assert len(history) == 1
    assert history.iloc[0]["Ejecución"] == "new"


def test_crest_uri_is_local_and_has_deterministic_fallback(tmp_path):
    dashboard = dashboard_module()
    first = dashboard.crest_data_uri("Equipo inventado", tmp_path)
    second = dashboard.crest_data_uri("Equipo inventado", tmp_path)
    assert first == second
    assert first.startswith("data:image/svg+xml;base64,")


def test_every_current_team_has_a_local_crest_file():
    from champions.data import CURRENT_TEAMS

    crest_dir = APP.parent / "assets" / "crests"
    mapping = json.loads((crest_dir / "team-crests.json").read_text(encoding="utf-8"))
    assert set(mapping) == set(CURRENT_TEAMS)
    assert all((crest_dir / filename).is_file() for filename in mapping.values())
    assert all(dashboard_module().crest_data_uri(team).startswith("data:image/png;base64,")
               for team in CURRENT_TEAMS)
    sources = json.loads((crest_dir / "sources.json").read_text(encoding="utf-8"))
    assert all(source["page"] and source["image"] for source in sources.values())


def test_dashboard_contains_only_the_three_requested_sections(root):
    snapshot(root)
    app = start()
    table = app.dataframe[0].value
    assert table.loc[0, "Equipo"] == "Real Madrid"
    assert table.loc[0, "Campeón"] == pytest.approx(20)
    assert table.columns[1] == "Campeón"
    assert table.columns[-1] == "Escudo"
    assert "Puestos 17–24" in table
    assert [item.value for item in app.subheader] == [
        "Probabilidades por equipo",
        "Clasificación esperada",
        "Evolución de la posición esperada",
    ]
    assert not app.title
    assert not app.metric
    assert not app.selectbox
    assert not app.multiselect
    assert not app.expander
    assert not app.button
    assert not app.json


def test_latest_snapshot_is_selected_automatically_and_history_is_kept(root):
    snapshot(root)
    snapshot(root, "run-2", day=2, probability=.3)
    snapshot(root, "run-3", season="2025/26", day=8)
    app = start()
    assert app.dataframe[0].value.set_index("Equipo").loc["Real Madrid", "Campeón"] == pytest.approx(30)
    assert app.get("plotly_chart")
    assert any("2 de 36" in item.value for item in app.warning)
    assert not app.selectbox


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
    snapshot(root, "run-2", day=2, probability=.4)
    app.run()
    assert app.dataframe[0].value.set_index("Equipo").loc["Real Madrid", "Campeón"] == pytest.approx(40)


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
    assert not app.exception
    assert not app.metric


def test_null_optional_metadata_does_not_break_dashboard(root):
    folder = snapshot(root)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    metadata["simulations"] = None
    metadata["known_results"] = "desconocido"
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    app = start()
    assert not app.exception
    assert not app.metric


def test_multiple_runs_same_day_and_absent_team(root):
    snapshot(root)
    folder = snapshot(root, "run-2", probability=.3)
    data = pd.read_csv(folder / "probabilities.csv")
    data[data.team == "Barcelona"].to_csv(folder / "probabilities.csv", index=False)
    app = start()
    assert app.dataframe[0].value["Equipo"].tolist() == ["Barcelona"]
    assert not app.selectbox


def test_offline_dashboard_and_local_crest_at_zero(root, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Dashboard attempted a network connection")

    snapshot(root, probability=0)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    app = start()
    table = app.dataframe[0].value.set_index("Equipo")
    assert table.loc["Real Madrid", "Campeón"] == 0
    assert table.loc["Real Madrid", "Escudo"].startswith("data:image/")
