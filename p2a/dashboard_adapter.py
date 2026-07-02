"""Compatibility module for the unified dashboard adapter."""

import sys

from p2a.dashboard_dynamic import adapter as _adapter

sys.modules[__name__] = _adapter
