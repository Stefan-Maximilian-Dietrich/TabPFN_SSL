import os

import numpy as np
import pytest


def _has_hf_token() -> bool:
    return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))


@pytest.mark.skipif(os.getenv("RUN_SLOW", "0") != "1", reason="Set RUN_SLOW=1 to run TabPFN fit test.")
@pytest.mark.skipif(not _has_hf_token(), reason="HF token missing (TabPFN v2.5 model download is gated).")
def test_tabpfn_minimal_fit_and_predict():
    from tabpfn import TabPFNClassifier

    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, size=30)

    clf = TabPFNClassifier(device="cpu")
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == (30,)