"""Compatibility module for the unified dashboard server."""

import sys

from p2a.dashboard_dynamic import server as _server

if __name__ == "__main__":
    raise SystemExit(_server.main())

sys.modules[__name__] = _server
