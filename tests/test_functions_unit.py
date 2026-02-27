# tests/test_functions_unit.py
import numpy as np

import scripts.functions as fun
import modules.data as ds


def test_upsample_returns_correct_sizes():
    data = ds.BreastCancer()()  
    n, m = 20, 10
    sampler = fun.upsample(n=n, m=m, Data=data)

    labeled, unlabeled, test = sampler(seed=0)

    assert len(labeled) == n
    assert len(unlabeled) == m
    assert len(test) == len(data) - n - m
    assert "target" in labeled.columns


def test_save_confusion_matrices_long_writes_csv(tmp_path):
    # Fake: 2 confusion matrices 2x2
    result = [
        np.array([[1, 2], [3, 4]]),
        np.array([[0, 1], [1, 0]]),
    ]
    csv_path = tmp_path / "out" / "cm.csv"

    fun.save_confusion_matrices_long(
        result=result,
        csv_path=str(csv_path),
        jobid="noid",
        rank=0,
        seed=123,
    )

    assert csv_path.exists()
    text = csv_path.read_text()
    assert "jobid" in text
    assert "cm_index" in text 