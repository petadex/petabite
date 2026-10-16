"""Tests for the small utility helpers.

The Registry tests that used to live here are gone with the registry itself:
it was generic machinery whose only remaining use on main was being re-exported
from `petabite.utils`, with no caller anywhere.
"""

from petabite.utils import set_seed


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    first = random.random()
    set_seed(123)
    second = random.random()

    assert first == second
