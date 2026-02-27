# tasks/local_smoke.py
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

import modules.data as ds
import scripts.functions as fun


class SklearnLogReg:
    name = "sklearn_logreg"

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, df):
        X = df.drop(columns=["target"]).to_numpy()
        y = df["target"].to_numpy()
        self.model.fit(X, y)
        return self

    def predict(self, df):
        X = df.drop(columns=["target"]).to_numpy()
        return self.model.predict(X)


class FirstDecision:
    def __init__(self, _Classifier):
        self.name = "first"

    def __call__(self, labeled, pseudo):
        return 0


Experiments = [
    {
        "n": 20,                    
        "m": 10,                    
        "Data": ds.BreastCancer(),  
        "Sampler": fun.upsample,
        "Evaluation": fun.confusion,
        "Classifier": SklearnLogReg(),
        "Decision": FirstDecision,
        "Predict": fun.predictor,
    }
]