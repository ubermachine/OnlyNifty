"""Nifty Institutional Trading Plan & Options Engine (JustNifty v2.0)"""
import os
__version__ = "2.0.0"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if "_PACKAGE_DIR" in locals() and _PACKAGE_DIR not in __path__:
    __path__.insert(0, _PACKAGE_DIR)

