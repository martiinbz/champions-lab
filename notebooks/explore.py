# %% [markdown]
# # Exploración reproducible de Champions Lab
# Ejecutar con Python o abrir como notebook por celdas en VS Code/Jupyter.
# Solo lee snapshots: no modifica resultados ni entrena con datos futuros.

# %%
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
runs = sorted((ROOT / "results/snapshots").glob("*/metadata.json"))
if not runs:
    raise SystemExit("Ejecuta champions update y champions simulate antes de explorar.")
selected = runs[-1]
metadata = json.loads(selected.read_text(encoding="utf-8"))
probabilities = pd.read_csv(selected.parent / "probabilities.csv")
coverage = pd.read_csv(ROOT / metadata["data_snapshot"] / "coverage.csv")

# %%
print(probabilities.sort_values("champion", ascending=False).to_string(index=False))
print(coverage.to_string(index=False))
print(json.dumps(metadata["evaluation"], indent=2, ensure_ascii=False))

# %%
probabilities.set_index("team")[["top8", "round16", "champion"]].sort_values("champion").plot.barh(
    figsize=(10, 14), title=f"{metadata['season']} · corte {metadata['cutoff']}"
)
