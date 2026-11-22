"""FastAPI surface for wildlife photo species evaluation."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .database import initialize
from .models import (
    BatchCreateRequest,
    CorrectionOut,
    CorrectionRequest,
    EvaluationOut,
)
from .service import ServiceError, apply_correction, create_batch, load_evaluation

DEFAULT_DB_PATH = "data/evaluations.sqlite3"


def _db_path() -> str:
    return os.environ.get("EVALUATION_DB", DEFAULT_DB_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize(_db_path())
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Wildlife Camera Evaluation API",
        version="1.0.0",
        description=(
            "Evaluate species predictions on camera-trap batches while excluding "
            "same-session photos from each other's neighbour search."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def db_path() -> str:
        return _db_path()

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": jsonable_encoder(exc.errors())},
        )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/batches",
        response_model=EvaluationOut,
        status_code=201,
        tags=["evaluation"],
        summary="Create a batch and evaluate it at version 1",
    )
    def create_batch_endpoint(
        payload: BatchCreateRequest, path: str = Depends(db_path)
    ) -> dict:
        return create_batch(path, payload)

    @app.get(
        "/batches/{batch_id}/evaluation",
        response_model=EvaluationOut,
        tags=["evaluation"],
        summary="Read the current committed evaluation",
    )
    def read_evaluation(batch_id: str, path: str = Depends(db_path)) -> dict:
        return load_evaluation(path, batch_id)

    @app.post(
        "/batches/{batch_id}/corrections",
        response_model=CorrectionOut,
        tags=["evaluation"],
        summary="Apply a versioned link correction and recompute affected photos",
    )
    def correct_batch(
        batch_id: str, payload: CorrectionRequest, path: str = Depends(db_path)
    ) -> dict:
        return apply_correction(path, batch_id, payload)

    return app


app = create_app()
