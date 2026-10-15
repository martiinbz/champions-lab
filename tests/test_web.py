import json
from pathlib import Path

import pandas as pd
import pytest

from champions import web

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root, run_id, cutoff, matchday, ranks, created):
    folder = root / "results" / "snapshots" / run_id
    folder.mkdir(parents=True)
    meta = {"run_id": run_id, "season": "2026/27", "matchday": matchday, "cutoff": cutoff,
            "created_at": created, "simulations": 100, "known_results": 0,
            "warnings": ["aviso </script><b>x</b>"], "data_hashes": {"fixtures.csv": run_id}}
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = [{"team": t, "champion": .5 / (i + 1), "top8": .5, "expected_rank": r, "expected_points": 10.0}
            for i, (t, r) in enumerate(ranks.items())]
    pd.DataFrame(rows).to_csv(folder / "probabilities.csv", index=False)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "template.html").write_text((ROOT / "web" / "template.html").read_text(encoding="utf-8"),
                                                    encoding="utf-8")
    return tmp_path


def test_empty_project_exports_an_empty_state_page(project):
    out = web.export(project)
    html = out.read_text(encoding="utf-8")
    assert '"empty":true' in html and web.PLACEHOLDER not in html


def test_headline_is_latest_cutoff_and_history_has_one_point_per_state(project):
    snapshot(project, "a", "2026-09-30T00:00:00+00:00", 2, {"X": 1.0, "Y": 2.0}, "2026-10-02T00:00:00+00:00")
    # Computed later but a retrospective reconstruction of an earlier cutoff.
    snapshot(project, "b", "2026-09-01T00:00:00+00:00", 0, {"X": 2.0, "Y": 1.0}, "2026-10-05T00:00:00+00:00")
    payload = web.build_payload(project)
    assert payload["run"]["run_id"] == "a"
    assert [p["matchday"] for p in payload["evolution"]] == [0, 2]
    assert payload["teams"][0]["team"] == "X"
    assert payload["matches"] == []  # no data snapshot available: reported, not fatal
    assert any("Partidos no disponibles" in issue for issue in payload["issues"])


def test_embedded_data_cannot_break_out_of_the_script_element(project):
    snapshot(project, "a", "2026-09-30T00:00:00+00:00", 2, {"X": 1.0}, "2026-10-02T00:00:00+00:00")
    html = web.export(project).read_text(encoding="utf-8")
    start = html.index("const DATA = ") + len("const DATA = ")
    block = html[start:html.index("</script>", start)]
    assert "</script" not in block and "<\\/script>" in block
    assert json.loads(block.strip().rstrip(";"))["run"]["run_id"] == "a"


def test_invalid_snapshot_is_reported_and_skipped(project):
    snapshot(project, "a", "2026-09-30T00:00:00+00:00", 2, {"X": 1.0}, "2026-10-02T00:00:00+00:00")
    broken = project / "results" / "snapshots" / "broken"
    broken.mkdir()
    (broken / "metadata.json").write_text("{}", encoding="utf-8")
    payload = web.build_payload(project)
    assert payload["run"]["run_id"] == "a" and any("broken" in i for i in payload["issues"])
