# tasks/local_smoke.py
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

import modules.data as ds
import scripts.functions as fun


class SklearnLogReg:
    """Minimaler Classifier-Wrapper, kompatibel mit deinem SSL-Code."""
    name = "sklearn_logreg"

    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

    def fit(self, df):
        X = df.drop(columns=["target"]).to_numpy()
        y = df["target"].to_numpy()
        self.model.fit(X, y)
        return self

    def predict(self, df):
        X = df.drop(columns=["target"]).to_numpy()
        return self.model.predict(X)


class FirstDecision:
    """Deterministische Decision: wählt immer das erste Pseudo-Sample."""
    def __init__(self, _Classifier):
        self.name = "first"

    def __call__(self, labeled, pseudo):
        return 0


# Ein einziges Mini-Experiment: sehr klein, läuft in Sekunden
Experiments = [
    {
        "n": 20,                    # labeled
        "m": 10,                    # unlabeled
        "Data": ds.BreastCancer(),  # lokal, schnell :contentReference[oaicite:7]{index=7}
        "Sampler": fun.upsample,
        "Evaluation": fun.confusion,
        "Classifier": SklearnLogReg(),
        "Decision": FirstDecision,
        "Predict": fun.predictor,
    }
]