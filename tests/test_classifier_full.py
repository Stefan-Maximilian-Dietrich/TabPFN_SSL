import pandas as pd
import numpy as np

import modules.classifier as clf_mod


def _make_df(n=10):
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.random.randint(0, 2, n),
    })


def test_logreg_fit_and_predict():
    model = clf_mod.SklearnLogReg()

    df = _make_df(20)

    model.fit(df)
    preds = model.predict(df)

    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_classifier_has_name():
    model = clf_mod.SklearnLogReg()
    assert isinstance(model.name, str)
    assert len(model.name) > 0