from cvnpy.api import lookup


def test_lookup_symbol_exists() -> None:
    assert callable(lookup)
