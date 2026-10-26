import unittest
from unittest.mock import MagicMock


if __name__ == '__main__':
    unittest.main()

from src.utils.decoding import get_decoding_strategy


# ---------------------------------------------------------------------------
# TC-18: test_get_decoding_strategy_factory
# ---------------------------------------------------------------------------
def test_get_decoding_strategy_one_shot():
    """Model without decoder._greedy returns 'one_shot'."""
    model = MagicMock(spec=[])  # no 'decoder' attribute
    strategy = get_decoding_strategy(model)
    assert strategy == "one_shot"


def test_get_decoding_strategy_stepwise():
    """Model with decoder._greedy attribute returns 'stepwise'."""
    model = MagicMock()
    model.decoder = MagicMock()
    model.decoder._greedy = True  # signals stepwise model
    strategy = get_decoding_strategy(model)
    assert strategy == "stepwise"
