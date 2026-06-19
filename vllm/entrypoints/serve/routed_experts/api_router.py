# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""REST API endpoints for global routed-experts statistics.

Exposes:
    GET  /v1/routed-experts/stats         — current aggregated stats
    POST /v1/routed-experts/stats/reset   — clear accumulated stats
    POST /v1/routed-experts/stats/disable — pause collection
    POST /v1/routed-experts/stats/enable  — resume collection

The router is only attached when ``enable_routed_experts_stats=True``
in the model config. All endpoints are unprotected by default (same
as other v1 endpoints).
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.core.sched.routed_experts_stats_collector import (
    aggregate_snapshots,
    snapshot_to_dict,
)

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/v1/routed-experts/stats")
async def get_routed_experts_stats(
    raw_request: Request,
    limit: int = Query(
        10,
        ge=1,
        le=100,
        description=(
            "Maximum number of entries in the "
            "most_activated_experts / least_activated_experts lists."
        ),
    ),
    include_zeros: bool = Query(
        False,
        description=(
            "If true, include experts with zero activations in the "
            "sorted lists (useful for debugging)."
        ),
    ),
):
    """Return current routed-experts statistics aggregated across all DP ranks."""
    client = engine_client(raw_request)
    snapshots = await client.get_routed_experts_stats()
    if snapshots is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "stats_disabled",
                    "message": (
                        "Routed-experts stats collection is not enabled. "
                        "Start the server with --enable-routed-experts-stats."
                    ),
                }
            },
        )

    aggregated = aggregate_snapshots(snapshots)
    return JSONResponse(content=snapshot_to_dict(
        aggregated, limit=limit, include_zeros=include_zeros
    ))


@router.post("/v1/routed-experts/stats/reset")
async def reset_routed_experts_stats(raw_request: Request):
    """Reset all accumulated routed-experts statistics."""
    client = engine_client(raw_request)
    await client.reset_routed_experts_stats()
    return JSONResponse(
        content={"status": "ok", "message": "Statistics reset"}
    )


@router.post("/v1/routed-experts/stats/disable")
async def disable_routed_experts_stats(raw_request: Request):
    """Disable routed-experts stats collection (keeps accumulated data)."""
    client = engine_client(raw_request)
    await client.set_routed_experts_stats_enabled(False)
    return JSONResponse(
        content={
            "status": "ok",
            "message": "Statistics collection disabled",
        }
    )


@router.post("/v1/routed-experts/stats/enable")
async def enable_routed_experts_stats(raw_request: Request):
    """Re-enable routed-experts stats collection."""
    client = engine_client(raw_request)
    await client.set_routed_experts_stats_enabled(True)
    return JSONResponse(
        content={
            "status": "ok",
            "message": "Statistics collection enabled",
        }
    )


def attach_router(app: FastAPI) -> None:
    """Attach the routed-experts stats router to the FastAPI app.

    Only attaches when ``enable_routed_experts_stats=True`` in the
    model config. Logs a warning when attached so operators know the
    endpoint is exposed.
    """
    args = getattr(app.state, "args", None)
    if args is None:
        return
    if not getattr(args, "enable_routed_experts_stats", False):
        return

    logger.warning_once(
        "Routed-experts stats API is enabled. "
        "Endpoints: GET /v1/routed-experts/stats, "
        "POST /v1/routed-experts/stats/{reset,disable,enable}. "
        "This should ONLY be used in trusted environments."
    )
    app.include_router(router)
