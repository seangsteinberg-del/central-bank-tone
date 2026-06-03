"""HTTP routers, one module per resource."""

from __future__ import annotations

from cbt_api.routes.speakers import router as speakers_router

__all__ = ["speakers_router"]
