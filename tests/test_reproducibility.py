import pandas as pd

import modules.data as ds
import scripts.functions as fun


def _fingerprint(df: pd.DataFrame) -> tuple:
    """
    Make a stable fingerprint of a split. We avoid relying on DataFrame index,
    because some samplers may reset/reorder indices.
    """
    # sort columns to be robust to column order
    cols = sorted(df.columns)
    # use values rounded a bit to avoid float noise
    vals = df[cols].to_numpy()
    # build a lightweight fingerprint: shape + per-column sums + target counts
    target_counts = tuple(df["target"].value_counts(dropna=False).sort_index().tolist())
    col_sums = tuple(vals.sum(axis=0).round(6).tolist())
    return (df.shape, target_counts, col_sums)


def test_sampler_reproducible_for_same_seed():
    data = ds.BreastCancer()()  # DataFrame with 'target'
    sampler = fun.upsample(n=20, m=10, Data=data)

    labeled1, unlabeled1, test1 = sampler(seed=0)
    labeled2, unlabeled2, test2 = sampler(seed=0)

    assert _fingerprint(labeled1) == _fingerprint(labeled2)
    assert _fingerprint(unlabeled1) == _fingerprint(unlabeled2)
    assert _fingerprint(test1) == _fingerprint(test2)