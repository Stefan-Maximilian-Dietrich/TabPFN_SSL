import numpy as np

import modules.data as ds


def test_breastcancer_dataframe_sanity():
    df = ds.BreastCancer()()

    assert "target" in df.columns
    assert len(df) > 0

    assert not df["target"].isna().any()

    X = df.drop(columns=["target"])
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes), "Non-numeric feature columns found."

    # no NaNs / inf
    Xv = X.to_numpy()
    assert np.isfinite(Xv).all(), "Found NaN/inf in features."