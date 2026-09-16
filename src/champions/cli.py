"""Command line entry point; no updates occur implicitly when opening Streamlit."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Champions Lab — actualización y simulación reproducible")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Carpeta raíz del proyecto")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("update", help="Descargar datos actuales e históricos")
    forecast = commands.add_parser("simulate", help="Guardar una simulación con los datos locales")
    forecast.add_argument("--cutoff", default=None, help="Fecha/hora ISO UTC; por defecto ahora")
    forecast.add_argument("--matchday", type=int, default=None)
    forecast.add_argument("--simulations", type=int, default=10000)
    forecast.add_argument("--seed", type=int, default=42)
    forecast.add_argument("--model", choices=["auto", "poisson", "baseline"], default="auto")
    forecast.add_argument("--skip-evaluation", action="store_true")
    forecast.add_argument("--knockout", type=Path, help="JSON del cuadro oficial y partidos de eliminatorias ya conocidos")
    replay = commands.add_parser("reproduce", help="Verificar hashes y reproducir una ejecución")
    replay.add_argument("run_id")
    commands.add_parser("list", help="Listar snapshots")
    args = parser.parse_args()
    from champions.pipeline import run, update_data, reproduce
    try:
        if args.command == "update":
            report = update_data(args.root.resolve())
            value = {k: report[k] for k in ("downloaded_at", "raw_dir", "fixtures", "history_matches", "latest_history")}
        elif args.command == "simulate":
            value = {"snapshot": str(run(args.root.resolve(), args.cutoff or pd.Timestamp.now(tz="UTC").isoformat(),
                                         args.matchday, args.simulations, args.seed,
                                         not args.skip_evaluation, args.model, args.knockout))}
        elif args.command == "reproduce":
            value = reproduce(args.root.resolve(), args.run_id)
        else:
            value = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((args.root / "results/snapshots").glob("*/metadata.json"))]
        print(json.dumps(value, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
