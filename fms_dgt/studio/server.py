# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""DiGiT Studio FastAPI application.

This module is intentionally thin: it owns HTTP concerns only (routing,
request validation, error codes).  All file I/O and data processing live
in utils.py so they can be tested and extended independently.
"""

# Standard
import contextlib
import os

# Third Party
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Local
from fms_dgt.studio import utils


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):
    if os.path.isdir(_STUDIO_DIST):
        application.mount("/", StaticFiles(directory=_STUDIO_DIST, html=True), name="static")
    yield


app = FastAPI(title="DiGiT Studio", lifespan=_lifespan)

# Seeded from env vars on import; overridden by configure() when called explicitly.
_OUTPUT_DIR: str = os.environ.get("DGT_OUTPUT_DIR", "")
_TELEMETRY_DIR: str | None = os.environ.get("DGT_TELEMETRY_DIR") or None

# Path to the bundled static frontend (studio/out/ after `next build`).
_STUDIO_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "studio", "dist")
)


def configure(output_dir: str, telemetry_dir: str | None) -> None:
    global _OUTPUT_DIR, _TELEMETRY_DIR
    _OUTPUT_DIR = os.path.abspath(output_dir)
    _TELEMETRY_DIR = os.path.abspath(telemetry_dir) if telemetry_dir else None


# ===================================================================================
#                               API ROUTES
# ===================================================================================
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {"outputDir": _OUTPUT_DIR}


@app.get("/api/data/runs")
def list_runs(path: str = Query(...)):
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail='Invalid "path" in request.')
    try:
        return utils.walk(path, "", path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch runs") from exc


@app.get("/api/data/run")
def get_run(path: str = Query(...)):
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail='Invalid "path" in request.')
    try:
        task_cards = utils.load_task_cards(path)
        task_card = task_cards[-1]
        log = utils.load_log(path)
        results = utils.load_results(path)
        data_points = utils.load_data_points(path)
        pid = utils.resolve_pid(task_card, results)

        generation_stats = None
        transformation_stats = None
        if _TELEMETRY_DIR and task_card.get("run_id"):
            generation_stats = utils.load_generation_stats(task_card["run_id"], _TELEMETRY_DIR)
            if not generation_stats:
                transformation_stats = utils.load_transformation_stats(
                    task_card["run_id"], _TELEMETRY_DIR
                )

        return {
            "card": task_card,
            "log": log,
            "result": results.get(pid),
            "datapoints": data_points,
            "generationStats": generation_stats,
            "transformationStats": transformation_stats,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read run data") from exc


@app.put("/api/data/run")
async def cancel_run(path: str = Query(...), request: Request = None):
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail='Invalid "path" in request.')
    body = await request.json()
    if body.get("action") == "cancel":
        try:
            utils.cancel_run(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to terminate the run") from exc
    return JSONResponse(content=None)


@app.get("/api/telemetry/tokens")
def token_usage(path: str = Query(...)):
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail='Invalid "path" in request.')
    if not _TELEMETRY_DIR:
        return JSONResponse(content=None)
    try:
        task_cards = utils.load_task_cards(path)
        task_card = task_cards[-1]
        if not task_card.get("run_id"):
            return JSONResponse(content=None)
        usage = utils.load_token_usage(
            task_card["run_id"], task_card.get("build_id"), _TELEMETRY_DIR
        )
        return JSONResponse(content=usage)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read token usage") from exc


# ===================================================================================
#                               DEV ENTRY POINT
# ===================================================================================
if __name__ == "__main__":
    load_dotenv()
    # Re-read after dotenv so .env values are picked up if not already set.
    configure(
        output_dir=os.environ.get("DGT_OUTPUT_DIR", "output"),
        telemetry_dir=os.environ.get("DGT_TELEMETRY_DIR", "telemetry"),
    )
    _dev = os.environ.get("DGT_STUDIO_MODE") == "dev"
    uvicorn.run(
        "fms_dgt.studio.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("DIGIT_STUDIO_PORT", 4720)),
        reload=_dev,
        reload_dirs=[os.path.dirname(__file__)] if _dev else None,
        log_level="info" if _dev else "error",
    )
