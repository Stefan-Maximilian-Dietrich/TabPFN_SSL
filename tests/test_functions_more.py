import numpy as np
import pandas as pd

import scripts.functions as fun


class DummyClassifier:
    name = "dummy"

    def fit(self, df):
        return self

    def predict(self, df):
        # predict alternating 0/1
        return np.array([i % 2 for i in range(len(df))])


def test_confusion_class_usage():
    """
    confusion is a class in this repo, not a plain function.
    We test it via its call pattern used in SSL.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    evaluator = fun.confusion  # class reference

    # In your code confusion is called like:
    # Evaluation(self.Classifier).run(...)
    # but internally it eventually computes a 2x2 matrix.
    cm = evaluator(DummyClassifier())(y_true, y_pred)

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