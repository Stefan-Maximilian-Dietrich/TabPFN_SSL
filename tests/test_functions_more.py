import numpy as np
import pandas as pd

import scripts.functions as fun


class DummyClassifier:
    name = "dummy"

    def fit(self, df):
        return self

    def predict(self, df):
        # Predict all zeros
        return np.zeros(len(df), dtype=int)


def test_confusion_returns_2x2():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    cm = fun.confusion(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_true)


def test_predictor_produces_predictions_length():
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3],
            "f2": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )
    clf = DummyClassifier()
    preds = fun.predictor(clf, df)
    assert len(preds) == len(df)