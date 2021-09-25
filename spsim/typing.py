from typing import NamedTuple
from pydantic import confloat, validator


class DefocusRange(NamedTuple):
    """defocus range, microns, positive is underfocus"""
    lower: confloat(gt=0)
    upper: confloat(gt=0)