import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    fetch_covtype,
    fetch_openml,
    make_circles,
    make_classification,
)

# ---------------------------------------------------------------------
# Basisklasse
# ---------------------------------------------------------------------

class BaseDataset:
    target_col = "target"

    def __init__(self, name: str):
        self.name = name
        self.n_instances = None
        self.n_predictors = None
        self.n_classes = None
        self._df = None

    def _load(self) -> pd.DataFrame:
        raise NotImplementedError

    def _update_stats(self, df: pd.DataFrame):
        """Setzt n_instances, n_predictors, n_classes basierend auf dem DataFrame."""
        self.n_instances = len(df)

        if self.target_col in df.columns:
            self.n_predictors = df.shape[1] - 1
            self.n_classes = int(pd.Series(df[self.target_col]).nunique())
        else:
            self.n_predictors = df.shape[1]
            self.n_classes = None

    def __call__(self) -> pd.DataFrame:
        if self._df is None:
            df = self._load()

            if self.target_col in df.columns:
                # 1. erst als Kategorie interpretieren
                df[self.target_col] = df[self.target_col].astype("category")
                # 2. Original-Kategorien speichern (falls man die später braucht)
                df.attrs["target_categories"] = list(df[self.target_col].cat.categories)
                # 3. dann in Integer-Codes 0,1,2,... umwandeln
                df[self.target_col] = df[self.target_col].cat.codes.astype(int)

            if "data_name" not in df.attrs:
                df.attrs["data_name"] = self.name

            self._update_stats(df)
            self._df = df

        return self._df

# ---------------------------------------------------------------------
# Konkrete Datensatz-Klassen
# ---------------------------------------------------------------------

class BreastCancer(BaseDataset):
    def __init__(self):
        super().__init__(name="BreastCancer")

    def _load(self) -> pd.DataFrame:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        df.attrs["formula"] = "target ~ " + " + ".join(data.feature_names)
        df.attrs["data_name"] = "breast_cancer"
        return df
    


class Iris(BaseDataset):
    def __init__(self):
        super().__init__(name="Iris")

    def _load(self) -> pd.DataFrame:
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        df.attrs["formula"] = "target ~ " + " + ".join(data.feature_names)
        df.attrs["data_name"] = "iris"
        return df


class Bank(BaseDataset):
    """
    Swiss banknotes (genuine vs. fake).
    """

    def __init__(self):
        super().__init__(name="Bank")

    def _load(self) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/EricBrownTTU/ISQS6350/main/swiss.csv"
        df = pd.read_csv(url)

        df = df.rename(columns={
            "note": "target",
            "length": "Length",
            "left_width": "Left",
            "right_width": "Right",
            "bottom_margin": "Bottom",
            "top_margin": "Top",
            "diag_length": "Diagonal",
        })

        cols = ["target", "Diagonal", "Bottom", "Length"]
        df = df[cols]

        df.attrs["formula"] = "target ~ Diagonal + Bottom + Length"
        df.attrs["data_name"] = "bank"
        return df


class MtcarsVS(BaseDataset):
    """
    mtcars mit target = vs, Features mpg, cyl, disp.
    """

    def __init__(self):
        super().__init__(name="Mtcars")

    def _load(self) -> pd.DataFrame:
        url = "https://www.swi-prolog.org/pack/file_details/mtx/data/mtcars.csv"
        df = pd.read_csv(url)

        df["target"] = df["vs"]
        df = df[["target", "mpg", "cyl", "disp"]]

        df.attrs["formula"] = "target ~ mpg + cyl + disp"
        df.attrs["data_name"] = "mtcars_vs"
        return df


class Cassini(BaseDataset):
    """
    Cassini-ähnlicher 3-Klassen-Datensatz in 2D.
    target ~ x1 + x2
    """

    def __init__(self, n_samples: int = 1000, random_state: int = 1):
        super().__init__(name="Cassini")
        self.n_samples = n_samples
        self.random_state = random_state

    def _load(self) -> pd.DataFrame:
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.0,
            random_state=self.random_state,
        )

        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y

        df = df[["target", "x1", "x2"]]
        df.attrs["formula"] = "target ~ x1 + x2"
        df.attrs["data_name"] = "cassini_like"
        return df


class Circle2D(BaseDataset):
    """
    Zwei-Kreis-Datensatz.
    target ~ x1 + x2
    """

    def __init__(
        self,
        n_samples: int = 2000,
        noise: float = 0.05,
        factor: float = 0.5,
        random_state: int = 6082000,
    ):
        super().__init__(name="Circle2d")
        self.n_samples = n_samples
        self.noise = noise
        self.factor = factor
        self.random_state = random_state

    def _load(self) -> pd.DataFrame:
        X, y = make_circles(
            n_samples=self.n_samples,
            noise=self.noise,
            factor=self.factor,
            random_state=self.random_state,
        )

        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y

        df = df[["target", "x1", "x2"]]
        df.attrs["formula"] = "target ~ x1 + x2"
        df.attrs["data_name"] = "circle_2d"
        return df


class Seeds(BaseDataset):
    """
    UCI Seeds Dataset.
    """

    def __init__(self):
        super().__init__(name="Seeds")

    def _load(self) -> pd.DataFrame:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        colnames = [
            "area",
            "perimeter",
            "compactness",
            "kernel_length",
            "kernel_width",
            "asymmetry_coefficient",
            "kernel_groove_length",
            "class",
        ]
        df = pd.read_csv(url, sep=r"\s+", header=None, names=colnames)
        df = df.dropna()

        df = df[[
            "class", "area", "perimeter", "compactness",
            "kernel_length", "kernel_width",
            "asymmetry_coefficient", "kernel_groove_length"
        ]]

        df = df.rename(columns={"class": "target"})

        df.attrs["formula"] = (
            "target ~ area + perimeter + compactness + kernel_length + "
            "kernel_width + asymmetry_coefficient + kernel_groove_length"
        )
        df.attrs["data_name"] = "seeds"
        return df


class Spirals(BaseDataset):
    """
    Two-spirals Datensatz.
    target ~ X1 + X2
    """

    def __init__(self, n_samples: int = 500, noise: float = 0.02, random_state: int = 0):
        super().__init__(name="Spirals")
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state

    def _load(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.random_state)
        n = self.n_samples // 2

        theta = np.linspace(0, 2 * np.pi, n)
        r = theta

        x1 = np.c_[r * np.cos(theta), r * np.sin(theta)]
        x2 = np.c_[-r * np.cos(theta), -r * np.sin(theta)]

        X = np.vstack([x1, x2])
        X += rng.normal(scale=self.noise, size=X.shape)

        y = np.array([0] * n + [1] * n)
        df = pd.DataFrame(X, columns=["X1", "X2"])
        df["target"] = y

        df = df[["target", "X1", "X2"]]
        df.attrs["formula"] = "target ~ X1 + X2"
        df.attrs["data_name"] = "spirals"
        return df


class Wine(BaseDataset):
    """
    UCI Wine (sklearn.load_wine), target + 13 Features.
    """

    def __init__(self):
        super().__init__(name="Wine")

    def _load(self) -> pd.DataFrame:
        data = load_wine()
        feature_names = list(data.feature_names)

        df = pd.DataFrame(data.data, columns=feature_names)
        df["target"] = data.target

        df = df[["target"] + feature_names]
        df.attrs["formula"] = "target ~ " + " + ".join(feature_names)
        df.attrs["data_name"] = "wine_task"
        return df
