import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def _split_X_y(df: pd.DataFrame, target_col: str = "target"):

    if target_col not in df.columns:
        raise ValueError(f"Zielspalte '{target_col}' nicht im DataFrame gefunden.")
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y


def _to_X(data, target_col: str = "target"):

    if isinstance(data, pd.DataFrame):
        if target_col in data.columns:
            data = data.drop(columns=[target_col])
        return data.values
    else:
        return np.asarray(data)


class TabPfnClassifier:
 

    def __init__(self, **kwargs):
        self.name = "TabPFNClassifier"
        self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        self.model.fit(X, y)
        return self

    def predict(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict(X)

    def predict_proba(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict_proba(X)


class NaiveBayesClassifier:


    def __init__(self, variant: str = "gaussian", **kwargs):
        self.name = "NaiveBayesClassifier"

        if variant == "gaussian":
            self.model = GaussianNB(**kwargs)
        elif variant == "multinomial":
            self.model = MultinomialNB(**kwargs)
        else:
            raise ValueError("variant muss 'gaussian' oder 'multinomial' sein")

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        self.model.fit(X, y)
        return self

    def predict(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict(X)

    def predict_proba(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict_proba(X)


class MultinomialLogitClassifier:


    def __init__(self, max_iter: int = 1000, **kwargs):
        self.name = "MultinomialLogitClassifier"
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            **kwargs,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model wurde noch nicht mit fit() trainiert.")

    def predict(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class SmallNNClassifier:


    def __init__(
        self,
        hidden_layer_sizes=(20, 20),
        activation="relu",
        max_iter=3000,
        **kwargs
    ):
        self.name = "SmallNNClassifier"
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            **kwargs,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model wurde noch nicht mit fit() trainiert.")

    def predict(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class SVMClassifier:


    def __init__(self, C: float = 1.0, gamma: str = "scale", probability: bool = True, **kwargs):
        self.name = "SVMClassifier"
        self.scaler = StandardScaler()
        # probability=True, damit predict_proba verfügbar ist
        self.model = SVC(C=C, gamma=gamma, probability=probability, **kwargs)
        self._fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model wurde noch nicht mit fit() trainiert.")

    def predict(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RandomForestCls:


    def __init__(
        self,
        n_estimators: int = 200,
        max_depth=None,
        random_state: int = 0,
        **kwargs
    ):
        self.name = "RandomForestCls"
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        self.model.fit(X, y)
        return self

    def predict(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict(X)

    def predict_proba(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict_proba(X)


class GradientBoostingCls:

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        **kwargs
    ):
        self.name = "GradientBoostingCls"
        self.model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **kwargs,
        )

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        self.model.fit(X, y)
        return self

    def predict(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict(X)

    def predict_proba(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict_proba(X)


class DecisionTreeCls:

    def __init__(self, max_depth=None, random_state: int = 0, **kwargs):
        self.name = "DecisionTreeCls"
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        self.model.fit(X, y)
        return self

    def predict(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict(X)

    def predict_proba(self, data, target_col: str = "target"):
        X = _to_X(data, target_col)
        return self.model.predict_proba(X)


class KNNClassifier:
 
    def __init__(self, n_neighbors: int = 2, **kwargs):
        self.name = "KNNClassifier"
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
        self._fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "target"):
        X, y = _split_X_y(df, target_col)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model wurde noch nicht mit fit() trainiert.")

    def predict(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, data, target_col: str = "target"):
        self._check_fitted()
        X = _to_X(data, target_col)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
