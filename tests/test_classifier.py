import inspect

import numpy as np
import pandas as pd
import pytest

import modules.classifier as clf_mod


def _make_df(n=30):
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "target": rng.randint(0, 2, size=n),
        }
    )


def _discover_classifier_classes():
    """
    Find all classes defined in modules.classifier that look like classifiers.
    We filter by presence of fit/predict attributes.
    """
    classes = []
    for _, obj in inspect.getmembers(clf_mod, inspect.isclass):
        # only classes defined in this module (avoid imported sklearn classes)
        if obj.__module__ != clf_mod.__name__:
            continue
        if hasattr(obj, "fit") and hasattr(obj, "predict"):
            classes.append(obj)
    classes.sort(key=lambda c: c.__name__)
    return classes


@pytest.mark.parametrize("cls", _discover_classifier_classes())
def test_classifier_init_fit_predict_smoke(cls):

    df = _make_df(40)

    try:
        model = cls()  # may fail if __init__ requires args
    except TypeError:
        pytest.skip(f"{cls.__name__} requires init args, skipping smoke test.")

    # name attribute is part of your pipeline (used for folder naming)
    assert hasattr(model, "name")
    assert isinstance(model.name, str)
    assert len(model.name) > 0

    model.fit(df)
    preds = model.predict(df)

    # allow list/np.array/pd.Series
    assert len(preds) == len(df)


def test_classifier_module_has_at_least_one_classifier():

    classes = _discover_classifier_classes()
    assert classes, "No classifier classes discovered in modules.classifier."