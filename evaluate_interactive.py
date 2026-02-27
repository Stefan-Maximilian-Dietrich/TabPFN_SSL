# evaluate_interactive.py

from __future__ import annotations
from pathlib import Path
import importlib.util
import sys


def _import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(__file__).resolve().parent
    eval_dir = root / "evaluation"

    summarize_path = eval_dir / "summarize_results.py"
    plots_path = eval_dir / "make_accuracy_plots.py"

    if not summarize_path.exists():
        raise FileNotFoundError(f"Missing: {summarize_path}")
    if not plots_path.exists():
        raise FileNotFoundError(f"Missing: {plots_path}")

    summarize = _import_from_path("summarize_results", summarize_path)
    plots = _import_from_path("make_accuracy_plots", plots_path)

    print("\nDo you want to create a tabular overview or plots?")
    print("  1) Tabular overview")
    print("  2) Plots")
    choice = input("Selection (1/2): ").strip()

    if choice == "1":
        summarize.main(results_dir="results", out_csv=str(eval_dir / "summary_results.csv"))
        return

    if choice != "2":
        print("Invalid selection.")
        sys.exit(1)

    plot_df = plots.build_plot_data("results")
    options = plots.list_possible_plots(plot_df)

    print("\nAvailable plots:\n")
    for _, r in options.iterrows():
        print(
            f"{int(r['nr'])}: dataset={r['dataset']} | "
            f"L={r['Labled data']} U={r['unlabled data']} | "
            f"classifyer={r['classifyer']}"
        )

    sel = input("\nWhich ones do you want to generate? (e.g. 'all' or '1-5,7,9'): ").strip()
    idxs = plots.parse_selection(sel, max_n=len(options))

    if not idxs:
        print("No valid selection.")
        return

    out_dir = eval_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plots.make_plots(plot_df, out_dir=str(out_dir), selections_1based=idxs)

    print(f"\nDone. Plots are located in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()