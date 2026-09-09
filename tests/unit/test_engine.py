import pytest

from tinytauk import TinyTAuK


def test_generate_fails_explicitly_in_v0() -> None:
    engine = TinyTAuK.from_pretrained()
    with pytest.raises(NotImplementedError, match="v0 is a scaffold"):
        engine.generate("Test")
