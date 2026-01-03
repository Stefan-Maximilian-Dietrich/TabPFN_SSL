import os
import glob
import math
import numpy as np
import pandas as pd
from collections import defaultdict


def confusion_matrix_from_long(df_idx):
    """
    df_idx: DataFrame mit Spalten [true, pred, count] nur für EINEN cm_index.
    Gibt eine (n_classes x n_classes)-Matrix zurück.
    """
    true_labels = sorted(df_idx["true"].unique())
    pred_labels = sorted(df_idx["pred"].unique())

    # Wir gehen davon aus, dass true/pred dieselben Klassen abdecken
    labels = sorted(set(true_labels) | set(pred_labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)

    cm = np.zeros((n, n), dtype=float)
    for _, row in df_idx.iterrows():
        t = label_to_idx[row["true"]]
        p = label_to_idx[row["pred"]]
        cm[t, p] += row["count"]

    return cm, labels


def accuracy_from_cm(cm):
    total = cm.sum()
    if total == 0:
        return float("nan")
    return np.trace(cm) / total


def f1_from_cm(cm):
    """
    Berechnet F1:
      - wenn 2x2: F1 für Klasse '1' (positive Klasse)
      - sonst: macro F1 über alle Klassen
    """
    n = cm.shape[0]
    if n == 2:
        # Binär: Klasse 1 als positiv
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    # Mehrklassen: macro F1
    f1s = []
    for k in range(n):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            f1_k = 0.0
        else:
            f1_k = 2 * prec * rec / (prec + rec)
        f1s.append(f1_k)

    return float(np.mean(f1s))


def aggregate_confusion_metrics(results_dir, metric="accuracy"):
    """
    results_dir: Pfad zum Ordner mit den CSV-Dateien.
    metric: "accuracy" oder "f1"

    Gibt ein DataFrame zurück mit:
      cm_index, mean_metric, std_metric, n_files
    und druckt eine kleine Übersicht.
    """

    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csv_files:
        print(f"Keine CSV-Dateien in {results_dir} gefunden.")
        return None

    if metric not in ("accuracy", "f1"):
        raise ValueError("metric muss 'accuracy' oder 'f1' sein.")

    values_per_index = defaultdict(list)

    for path in csv_files:
        df = pd.read_csv(path)

        if not {"cm_index", "true", "pred", "count"}.issubset(df.columns):
            print(f"Überspringe {path}, Spalten nicht vollständig.")
            continue

        # Pro Datei über alle cm_index iterieren
        for cm_idx, df_idx in df.groupby("cm_index"):
            cm, labels = confusion_matrix_from_long(df_idx)

            if metric == "accuracy":
                val = accuracy_from_cm(cm)
            else:  # "f1"
                val = f1_from_cm(cm)

            values_per_index[int(cm_idx)].append(val)

    # Zusammenfassen
    rows = []
    for cm_idx in sorted(values_per_index.keys()):
        vals = np.array(values_per_index[cm_idx], dtype=float)
        # Filtere NaN raus, falls vorhanden
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            mean_val = float("nan")
            std_val = float("nan")
        else:
            mean_val = float(vals.mean())
            std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        rows.append(
            {
                "cm_index": cm_idx,
                f"mean_{metric}": mean_val,
                f"std_{metric}": std_val,
                "n_files": len(values_per_index[cm_idx]),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("cm_index")

    print(f"\nAggregierte {metric}-Werte pro cm_index über {len(csv_files)} Dateien:\n")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Pfad zum Ordner mit den Confusion-CSV-Dateien.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1"],
        help="Welcher Score soll berechnet werden?",
    )
    args = parser.parse_args()

    aggregate_confusion_metrics(args.results_dir, metric=args.metric)
