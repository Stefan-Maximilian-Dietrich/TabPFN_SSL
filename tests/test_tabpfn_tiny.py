import numpy as np
import pytest

# Wenn tabpfn nicht installiert ist: Test wird sauber übersprungen (nicht failen)
pytest.importorskip("tabpfn")


@pytest.mark.tabpfn
def test_tabpfn_tiny_fit_predict_cpu():
    """
    Level-2 Test: Mini Forward Test (CPU)
    Prüft:
      - TabPFN Wrapper kann initialisiert werden
      - fit() läuft auf tiny dataset
      - predict() liefert richtige Länge
      - predict_proba() hat richtige Form und Wertebereich
    """

    # Import erst hier, damit Importfehler sauber als skip erscheinen
    import modules.data as ds
    from modules.classifier import TabPfnClassifier  # dein Wrapper

    # Tiny Dataset (lokal, sklearn) -> BreastCancer ist bei dir sehr schnell verfügbar
    df = ds.BreastCancer()()

    # sehr klein halten, damit es schnell bleibt
    df_small = df.sample(n=30, random_state=0).reset_index(drop=True)

    clf = TabPfnClassifier()
    clf.fit(df_small)

    preds = clf.predict(df_small)
    assert len(preds) == len(df_small)

    # Optional: check value range (binary case)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = clf.predict_proba(df_small)
    assert proba.shape[0] == len(df_small)
    assert proba.ndim == 2
    # Wahrscheinlichkeiten im [0,1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)