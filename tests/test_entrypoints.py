from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_new_simulation_command_forwards_user_count_and_seed():
    script = ROOT / "Nueva-simulacion.ps1"
    assert script.exists()
    contents = script.read_text(encoding="utf-8")
    assert "[long]$Simulaciones" in contents
    assert "--simulations $Simulaciones" in contents
    assert "--seed $Semilla" in contents
    assert "champions.cli simulate" in contents
