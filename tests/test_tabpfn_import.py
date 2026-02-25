import pytest


def test_tabpfn_import_and_instantiation():
    """
    This test ensures that TabPFN is properly installed and can be imported.
    It will fail if tabpfn is missing or broken.
    """
    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        pytest.fail(f"TabPFN import failed: {e}")

    # Try minimal instantiation (no training)
    try:
        model = TabPFNClassifier(device="cpu")
    except Exception as e:
        pytest.fail(f"TabPFN instantiation failed: {e}")