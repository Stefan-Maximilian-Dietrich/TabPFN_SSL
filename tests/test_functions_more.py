import numpy as np
import pandas as pd

import scripts.functions as fun


class DummyClassifier:
    name = "dummy"

    def fit(self, df):
        return self

    def predict(self, df):
        # deterministic 0/1 pattern
        return np.array([i % 2 for i in range(len(df))], dtype=int)


def _make_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, n),
            "f2": np.linspace(1.0, 2.0, n),
            "target": np.array([0, 1] * (n // 2) + ([0] if n % 2 else []), dtype=int),
        }
    )


def test_confusion_class_usage_dataframe_inputs():
    labeled = _make_df(6)
    test = _make_df(6)

    cm = fun.confusion(DummyClassifier())(labeled, test)

    assert cm.shape == (2, 2)
    assert cm.sum() == len(test)


def test_predictor_class_usage_dataframe_inputs():
    """
    predictor in this repo expects (labeled, unlabeled) DataFrames.
    It fits on labeled and predicts on unlabeled.
    """
    labeled = _make_df(6)
    unlabeled = _make_df(4)

    preds = fun.predictor(DummyClassifier())(labeled, unlabeled)

    assert len(preds) == len(unlabeled)
    assert set(np.unique(preds)).issubset({0, 1})