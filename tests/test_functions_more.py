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


def test_predictor_returns_dataframe_with_expected_shape_and_columns():
    labeled = _make_df(6)
    unlabeled = _make_df(4)

    out = fun.predictor(DummyClassifier())(labeled, unlabeled)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(unlabeled)

    # should contain at least the original columns
    for col in unlabeled.columns:
        assert col in out.columns