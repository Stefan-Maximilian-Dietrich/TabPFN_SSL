# evaluation/make_accuracy_plots.py

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def _prefix_before_underscore(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name


def _try_parse_trailing_int(name: str) -> str:
    m = re.search(r"_([0-9]+)$", name)
    return m.group(1) if m else name


def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if name.startswith(prefix) else name


def build_plot_data(results_dir: str = "results") -> pd.DataFrame:

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"results_dir not found: {results_path.resolve()}")

    rows = []

    for csv_path in results_path.rglob("ID_*.csv"):
        rel = csv_path.relative_to(results_path)
        parts = rel.parts
        if len(parts) < 5:
            continue

        dataset_folder = parts[0]
        unlabeled_folder = parts[1]
        classifier_folder = parts[2]
        decision_folder = parts[3]

        dataset = _prefix_before_underscore(dataset_folder)
        labeled = _try_parse_trailing_int(dataset_folder)
        unlabeled = _try_parse_trailing_int(unlabeled_folder)
        classifier = _strip_prefix(classifier_folder, "classifier_")
        decision = _strip_prefix(decision_folder, "decision_")

        df = pd.read_csv(csv_path)

        required = {"seed", "cm_index", "true", "pred", "count"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df["cm_index"] = pd.to_numeric(df["cm_index"], errors="coerce")
        df["true"] = pd.to_numeric(df["true"], errors="coerce")
        df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["seed", "cm_index", "true", "pred"])
        df["seed"] = df["seed"].astype(int)
        df["cm_index"] = df["cm_index"].astype(int)

        df["_correct"] = df["true"] == df["pred"]

        correct = df[df["_correct"]].groupby(["seed", "cm_index"])["count"].sum()
        total = df.groupby(["seed", "cm_index"])["count"].sum()
        acc = (correct / total).reset_index(name="acc")

        # shift iteration so it starts at 0 per seed
        acc["iteration"] = acc["cm_index"] - acc.groupby("seed")["cm_index"].transform("min")
        acc = acc.sort_values(["seed", "iteration"])

        for _, r in acc.iterrows():
            rows.append({
                "dataset": dataset,
                "Labled data": labeled,
                "unlabled data": unlabeled,
                "classifyer": classifier,
                "decision funktion": decision,
                "seed": int(r["seed"]),
                "iteration": int(r["iteration"]),
                "acc": float(r["acc"]),
            })

    if not rows:
        raise RuntimeError(f"No ID_*.csv found under {results_path.resolve()}")

    return pd.DataFrame(rows)


def list_possible_plots(plot_df: pd.DataFrame) -> pd.DataFrame:
    groups = (
        plot_df[["dataset", "Labled data", "unlabled data", "classifyer"]]
        .drop_duplicates()
        .sort_values(["dataset", "Labled data", "unlabled data", "classifyer"])
        .reset_index(drop=True)
    )
    groups.insert(0, "nr", range(1, len(groups) + 1))
    return groups


def parse_selection(text: str, max_n: int) -> list[int]:
    """
    Accepts:
    - "all"
    - "1,2,5"
    - "1-4"
    - "1-3,7,10-12"
    Returns 1-based indices.
    """
    t = text.strip().lower()
    if t in {"all", "a", "*"}:
        return list(range(1, max_n + 1))

    nums: set[int] = set()
    for part in t.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            lo, hi = min(lo, hi), max(lo, hi)
            for k in range(lo, hi + 1):
                nums.add(k)
        else:
            nums.add(int(part))

    return [n for n in sorted(nums) if 1 <= n <= max_n]


def make_plots(
    plot_df: pd.DataFrame,
    out_dir: str = "evaluation/plots",
    selections_1based: list[int] | None = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    groups = list_possible_plots(plot_df)

    if selections_1based is None:
        selected = groups
    else:
        # convert 1-based -> 0-based
        idx0 = [i - 1 for i in selections_1based]
        selected = groups.iloc[idx0].reset_index(drop=True)

    for _, g in selected.iterrows():
        dataset = g["dataset"]
        labeled = g["Labled data"]
        unlabeled = g["unlabled data"]
        classifier = g["classifyer"]

        sub = plot_df[
            (plot_df["dataset"] == dataset)
            & (plot_df["Labled data"] == labeled)
            & (plot_df["unlabled data"] == unlabeled)
            & (plot_df["classifyer"] == classifier)
        ].copy()

        # mean across seeds per (decision, iteration)
        mean_df = (
            sub.groupby(["decision funktion", "iteration"], as_index=False)
               .agg(acc=("acc", "mean"))
               .sort_values(["decision funktion", "iteration"])
        )

        plt.figure()
        for decision, dsub in mean_df.groupby("decision funktion"):
            plt.plot(dsub["iteration"], dsub["acc"], label=str(decision))

        title = f"{dataset} | L={labeled} U={unlabeled} | {classifier}"
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.legend()

        # annotate dataset / L / U / classifier inside plot
        plt.text(
            0.02, 0.02,
            f"dataset={dataset}\nL={labeled}  U={unlabeled}\nclassifier={classifier}",
            transform=plt.gca().transAxes
        )

        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{dataset}_L{labeled}_U{unlabeled}_{classifier}")
        file_path = out_path / f"{safe_name}.png"
        plt.tight_layout()
        plt.savefig(file_path, dpi=200)
        plt.close()

        print(f"Saved plot: {file_path.resolve()}")