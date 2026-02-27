import inspect
from typing import Callable, Dict, List, Type

import pandas as pd
import pytest

import modules.data as ds


def _patch_network_read_csv(monkeypatch):
    """
    Patch pd.read_csv for the three network datasets:
      - Bank (swiss.csv)
      - MtcarsVS (mtcars.csv)
      - Seeds (seeds_dataset.txt)
    so tests are stable offline/in CI.
    """
    real_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        path_str = str(path)

        if "swiss.csv" in path_str:
            return pd.DataFrame(
                {
                    "note": [0, 1, 0, 1],
                    "length": [10.0, 10.1, 9.9, 10.2],
                    "left_width": [5.0, 5.1, 4.9, 5.2],
                    "right_width": [5.0, 5.1, 4.9, 5.2],
                    "bottom_margin": [2.0, 2.1, 1.9, 2.2],
                    "top_margin": [3.0, 3.1, 2.9, 3.2],
                    "diag_length": [12.0, 12.1, 11.9, 12.2],
                }
            )

        if "mtcars.csv" in path_str:
            return pd.DataFrame(
                {
                    "vs": [0, 1, 0, 1],
                    "mpg": [21.0, 22.0, 18.0, 30.0],
                    "cyl": [6, 4, 8, 4],
                    "disp": [160.0, 108.0, 360.0, 75.0],
                }
            )

        if "seeds_dataset.txt" in path_str:
            return pd.DataFrame(
                {
                    "area": [10.0, 11.0, 12.0],
                    "perimeter": [14.0, 15.0, 16.0],
                    "compactness": [0.8, 0.9, 0.85],
                    "kernel_length": [5.0, 5.1, 5.2],
                    "kernel_width": [3.0, 3.1, 3.2],
                    "asymmetry_coefficient": [2.0, 2.1, 2.2],
                    "kernel_groove_length": [4.0, 4.1, 4.2],
                    "class": [1, 2, 1],
                }
            )

        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)


def _discover_dataset_classes() -> List[Type[ds.BaseDataset]]:
    """
    Discover all concrete BaseDataset subclasses defined in modules.data.
    """
    out: List[Type[ds.BaseDataset]] = []
    for _, obj in inspect.getmembers(ds, inspect.isclass):
        if obj is ds.BaseDataset:
            continue
        if issubclass(obj, ds.BaseDataset):
            out.append(obj)
    # stable order
    out.sort(key=lambda c: c.__name__)
    return out



_FACTORIES: Dict[str, Callable[[], ds.BaseDataset]] = {
    "Cassini": lambda: ds.Cassini(n_samples=80, random_state=1),
    "Circle2D": lambda: ds.Circle2D(n_samples=80, noise=0.05, factor=0.5, random_state=1),
    "Spirals": lambda: ds.Spirals(n_samples=80, noise=0.02, random_state=1),
}


def _make_instance(cls: Type[ds.BaseDataset]) -> ds.BaseDataset:
    if cls.__name__ in _FACTORIES:
        return _FACTORIES[cls.__name__]()
    return cls()  # type: ignore[call-arg]


@pytest.mark.parametrize("dataset_cls", _discover_dataset_classes())
def test_all_datasets_init_and_load(monkeypatch, dataset_cls):
    """
    Generic smoke test for ALL dataset classes:
      - __init__ works
      - __call__ triggers _load and returns a non-empty DataFrame
      - target exists and is int-coded (because BaseDataset.__call__ converts category->codes)
      - stats fields are set (n_instances, n_predictors, n_classes)
    """
    _patch_network_read_csv(monkeypatch)

    d = _make_instance(dataset_cls)
    df = d()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # BaseDataset contract: if target exists, it should be int after conversion.
    assert d.target_col in df.columns
    assert pd.api.types.is_integer_dtype(df[d.target_col])

    # stats should be set
    assert d.n_instances == len(df)
    assert d.n_predictors is not None
    assert d.n_classes is not None
    assert d.n_classes >= 2

    # attrs should contain data_name
    assert "data_name" in df.attrs