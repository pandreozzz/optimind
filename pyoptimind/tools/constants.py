"""Physical constants used across the project."""

from __future__ import annotations

from typing import Final

__all__ = ["CPAIR", "GCNST"]

# Specific heat capacity of air at constant pressure [J kg^-1 K^-1]
CPAIR: Final[float] = 1.0e3

# Gravitational acceleration [m s^-2]
GCNST: Final[float] = 9.81

# Gas constant for dry air
R_DRYAIR = 287.05 # J/Kg K
