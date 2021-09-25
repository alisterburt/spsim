import pytest

from spsim.typing import DefocusRange


def test_defocus_range_instantiation_valid():
    defocus_range = DefocusRange(0.5, 8.5)
    assert isinstance(defocus_range, DefocusRange)
    assert defocus_range.lower == 0.5
    assert defocus_range.upper == 8.5
