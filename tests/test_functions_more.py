import numpy as np
import pandas as pd

import scripts.functions as fun


class DummyClassifier:
    name = "dummy"

    def fit(self, df):
        # df is a DataFrame with features + "target"
        return self

    def predict(self, df):
        # df is a DataFrame with features + "target"
        # Return a deterministic prediction vector
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
    """
    In this repo, confusion is a callable object that:
      - fits the model on 'labled' (DataFrame)
      - predicts on 'test' (DataFrame)
      - computes sklearn confusion_matrix(test["target"], y_pred)
    So we must pass DataFrames, not y arrays.
    """
    labeled = _make_df(6)
    test = _make_df(6)

    cm = fun.confusion(DummyClassifier())(labeled, test)

    assert cm.shape == (2, 2)
    assert cm.sum() == len(test)


def test_predictor_class_usage_dataframe_input():
    """
    predictor is also a class/callable in this repo.
    It should call model.predict(df) and return a prediction vector.
    """
    df = _make_df(5)

    preds = fun.predictor(DummyClassifier())(df)

    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})