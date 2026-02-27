# evaluation/summarize_results.py

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd


def _prefix_before_underscore(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name


def _try_parse_trailing_int(name: str) -> str:
    m = re.search(r"_([0-9]+)$", name)
    return m.group(1) if m else name


def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if name.startswith(prefix) else name


def summarize_one_id_csv(path: Path) -> dict:
    df = pd.read_csv(path)

    required = {"cm_index", "true", "pred", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df["cm_index"] = pd.to_numeric(df["cm_index"], errors="coerce")
    df["true"] = pd.to_numeric(df["true"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["cm_index", "true", "pred"])

    if df.empty:
        return {"acc_start": float("nan"), "acc_max": float("nan"), "acc_end": float("nan")}

    df["_is_correct"] = df["true"] == df["pred"]

    correct = df.loc[df["_is_correct"]].groupby("cm_index")["count"].sum()
    total = df.groupby("cm_index")["count"].sum()
    acc = (correct / total).sort_index()

    return {
        "acc_start": float(acc.iloc[0]),
        "acc_max": float(acc.max()),
        "acc_end": float(acc.iloc[-1]),
    }


def _union_seed_sets(series: pd.Series) -> int:
    all_seeds = set()
    for s in series:
        if isinstance(s, set):
            all_seeds |= s
    return int(len(all_seeds))


def main(results_dir: str = "results", out_csv: str = "evaluation/summary_results.csv") -> None:
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

        metrics = summarize_one_id_csv(csv_path)

        # unique seeds in this file
        df_seed = pd.read_csv(csv_path, usecols=lambda c: c.strip() == "seed")
        df_seed["seed"] = pd.to_numeric(df_seed["seed"], errors="coerce")
        seeds_in_file = set(df_seed["seed"].dropna().astype(int).tolist())

        rows.append({
            "dataset": dataset,
            "Labled data": labeled,
            "unlabled data": unlabeled,
            "classifyer": classifier,
            "decision funktion": decision,
            "seeds_in_file": seeds_in_file,
            **metrics,
        })

    if not rows:
        raise RuntimeError(f"No ID_*.csv found under {results_path.resolve()}")

    df = pd.DataFrame(rows)

    group_cols = ["dataset", "Labled data", "unlabled data", "classifyer", "decision funktion"]

    agg = (
        df.groupby(group_cols, as_index=False)
          .agg(
              **{
                  "seeds tested": ("seeds_in_file", _union_seed_sets),
                  "accuracy at start": ("acc_start", "mean"),
                  "maximum accuracy": ("acc_max", "mean"),
                  "accuracy at end": ("acc_end", "mean"),
              }
          )
          .sort_values(group_cols)
          .reset_index(drop=True)
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)

    print("\nSummary:\n")
    for _, r in agg.iterrows():
        print(f"dataset:            {r['dataset']}")
        print(f"Labled data:        {r['Labled data']}")
        print(f"unlabled data:      {r['unlabled data']}")
        print(f"classifyer:         {r['classifyer']}")
        print(f"decision funktion:  {r['decision funktion']}")
        print(f"seeds tested:       {int(r['seeds tested'])}")
        print(f"accuracy at start:  {r['accuracy at start']:.6f}")
        print(f"maximum accuracy:   {r['maximum accuracy']:.6f}")
        print(f"accuracy at end:    {r['accuracy at end']:.6f}\n")

    print(f"Saved CSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()