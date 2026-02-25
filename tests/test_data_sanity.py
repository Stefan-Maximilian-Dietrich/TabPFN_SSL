import numpy as np

import modules.data as ds


def test_breastcancer_dataframe_sanity():
    df = ds.BreastCancer()()

    # Basic schema expectations
    assert "target" in df.columns
    assert len(df) > 0

    # target should not contain NaNs
    assert not df["target"].isna().any()

    # features: everything except target should be numeric and finite
    X = df.drop(columns=["target"])
    # numeric types
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes), "Non-numeric feature columns found."

    # no NaNs / inf
    Xv = X.to_numpy()
    assert np.isfinite(Xv).all(), "Found NaN/inf in features."