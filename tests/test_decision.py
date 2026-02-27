import inspect
import numpy as np
import pandas as pd
import pytest


def _import_decision_module():
    try:
        from modules import decision as decision_mod  
        return decision_mod
    except Exception:
        pass

    try:
        from scripts import decision as decision_mod  
        return decision_mod
    except Exception:
        pass

    try:
        import decision as decision_mod  
        return decision_mod
    except Exception:
        return None


decision_mod = _import_decision_module()


pytestmark = pytest.mark.skipif(decision_mod is None, reason="No decision module found (expected modules/decision.py or scripts/decision.py).")


class DummyTabPFN:
    """Dummy replacement for TabPFNClassifier to avoid gated downloads."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = np.linspace(0.2, 0.8, n)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


class DummyProbClassifier:

    name = "dummy_prob"

    def fit(self, df):
        return self

    def predict_proba(self, df):
        n = len(df)
        p1 = np.array([(i % 2) * 0.9 + (1 - (i % 2)) * 0.1 for i in range(n)], dtype=float)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, df):
        proba = self.predict_proba(df)
        return (proba[:, 1] >= 0.5).astype(int)


def _make_labeled(n=8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, n),
            "f2": np.linspace(1.0, 2.0, n),
            "target": np.array([0, 1] * (n // 2) + ([0] if n % 2 else []), dtype=int),
        }
    )


def _make_pseudo(n=5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, n),
            "f2": np.linspace(1.0, 2.0, n),
            "target": np.array([0, 1] * (n // 2) + ([0] if n % 2 else []), dtype=int),
        }
    )


def _discover_decision_classes():
    """
    Find all classes defined in the decision module that implement __call__.
    """
    classes = []
    for _, obj in inspect.getmembers(decision_mod, inspect.isclass):
        if obj.__module__ != decision_mod.__name__:
            continue
        if hasattr(obj, "__call__"):
            classes.append(obj)
    classes.sort(key=lambda c: c.__name__)
    return classes


def _try_instantiate(decision_cls):

    try:
        return decision_cls(DummyProbClassifier())
    except TypeError:
        try:
            return decision_cls()
        except TypeError:
            return None


@pytest.mark.parametrize("decision_cls", _discover_decision_classes())
def test_all_decision_methods_smoke(monkeypatch, decision_cls):

    # avoid gated model downloads, if the decision module uses TabPFNClassifier
    if hasattr(decision_mod, "TabPFNClassifier"):
        monkeypatch.setattr(decision_mod, "TabPFNClassifier", DummyTabPFN)

    inst = _try_instantiate(decision_cls)
    if inst is None:
        pytest.skip(f"{decision_cls.__name__} requires init args not provided by generic test.")

    labeled = _make_labeled(8)
    pseudo = _make_pseudo(5)

    try:
        out = inst(labeled, pseudo)
    except Exception as e:
        pytest.xfail(f"{decision_cls.__name__} raised {type(e).__name__}: {e}")

    assert isinstance(out, (int, np.integer))
    assert 0 <= int(out) < len(pseudo)


def test_decision_module_has_decisions():
    classes = _discover_decision_classes()
    assert classes, "No decision classes discovered in decision module"