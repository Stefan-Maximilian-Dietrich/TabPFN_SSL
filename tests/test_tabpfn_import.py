import pytest


def test_tabpfn_import_and_instantiation():
    """
    Ensure TabPFN can be imported and instantiated.
    """

    try:
        from tabpfn import DOES_NOT_EXIST
    except Exception as e:
        pytest.fail(f"TabPFN import failed: {e}")

    try:
        _ = TabPFNClassifier(device="cpu")
    except Exception as e:
        pytest.fail(f"TabPFN instantiation failed: {e}")