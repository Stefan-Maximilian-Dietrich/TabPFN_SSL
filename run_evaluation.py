from __future__ import annotations
import subprocess
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "Results"
EVAL_DIR = REPO_ROOT / "evaluation"
OUT_ROOT = EVAL_DIR / "outputs"
R_WRAPPER = EVAL_DIR / "run_r_analysis.R"


def choose_from_list(options: list[str], prompt: str) -> str:
    if not options:
        raise RuntimeError("Keine Optionen gefunden.")
    print("\n" + prompt)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        s = input("Nummer: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Bitte eine gültige Nummer eingeben.")


def pick_results_path() -> Path:
    if not RESULTS_ROOT.exists():
        raise RuntimeError(f"Results-Ordner nicht gefunden: {RESULTS_ROOT}")

    datasets = sorted([p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()])
    ds = choose_from_list(datasets, "Welches Dataset / Results-Verzeichnis willst du analysieren?")
    ds_path = RESULTS_ROOT / ds

    unlabeled = sorted([p.name for p in ds_path.iterdir()
                        if p.is_dir() and p.name.startswith("unlabeled_")])
    if unlabeled:
        choice = input("\nOptional: nur ein bestimmtes 'unlabeled_*' analysieren? (j/n): ").strip().lower()
        if choice == "j":
            ul = choose_from_list(unlabeled, "Welches unlabeled_*?")
            return ds_path / ul

    return ds_path


def run_r(action: str, base_path: Path, metric: str, show_sd: bool) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"{ts}_{action}_{metric}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "Rscript",
        str(R_WRAPPER),
        action,
        str(base_path),
        metric,
        str(out_dir),
        "1" if show_sd else "0",
    ]

    print("\nStarte R Analyse:\n" + " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Rscript failed with code {proc.returncode}")

    print(f"\nFertig. Outputs liegen hier:\n  {out_dir}")
    return out_dir


def main():
    actions = {
        "tables": "Intermediate tables (pro Ordner) als CSV speichern",
        "long":   "res_long als CSV speichern",
        "plots":  "Plots (PDF) + res_long.csv speichern",
        "all":    "Alles: tables + res_long + plots",
    }

    action = choose_from_list(list(actions.keys()), "Welche Aktion möchtest du ausführen?")
    print(f"→ {action}: {actions[action]}")

    metric = choose_from_list(["accuracy", "f1"], "Welche Metrik?")

    base_path = pick_results_path()
    print(f"\nAnalysiere Pfad:\n  {base_path}")

    show_sd = False
    if action in ("plots", "all"):
        show_sd = (input("Standardabweichungs-Band im Plot? (j/n): ").strip().lower() == "j")

    run_r(action=action, base_path=base_path, metric=metric, show_sd=show_sd)


if __name__ == "__main__":
    main()
