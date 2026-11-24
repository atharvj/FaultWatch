"""Local telemetry API and RUL prediction service."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParameter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from faultwatch.cache import CACHE_ENV, PredictionCache, prediction_key
from faultwatch.data import OPERATING_SETTINGS, SENSORS, TELEMETRY_COLUMNS
from faultwatch.db import DATABASE_ENV, connect_database, recent_readings, store_reading, telemetry
from faultwatch.features import WINDOW_SIZE
from faultwatch.inference import MODEL_VERSION, load_predictor
from faultwatch.metrics import Metrics, RequestMetrics

logger = logging.getLogger(__name__)
MAX_KEY = 2_147_483_647
MODEL_ENV = "FAULTWATCH_MODEL_DIR"
DEFAULT_MODEL_DIR = Path("artifacts") / MODEL_VERSION

PositiveKey = Annotated[int, Field(strict=True, ge=1, le=MAX_KEY)]
Reading = Annotated[float, Field(strict=True, allow_inf_nan=False)]
EnginePath = Annotated[int, PathParameter(ge=1, le=MAX_KEY)]


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine_id: PositiveKey
    cycle: PositiveKey
    settings: Annotated[
        list[Reading], Field(min_length=len(OPERATING_SETTINGS), max_length=len(OPERATING_SETTINGS))
    ]
    sensors: Annotated[list[Reading], Field(min_length=len(SENSORS), max_length=len(SENSORS))]


class StoredReading(TelemetryEvent):
    received_at: datetime


class IngestResult(BaseModel):
    engine_id: int
    cycle: int
    status: Literal["created", "duplicate"]


class EngineSummary(BaseModel):
    engine_id: int
    latest_cycle: int


class Prediction(BaseModel):
    engine_id: int
    cycle: int
    status: Literal["ready", "insufficient_history"]
    available_cycles: int
    required_cycles: int = WINDOW_SIZE
    predicted_rul_cycles: float | None = None
    model_version: str


def _database(request: Request) -> Engine:
    engine = request.app.state.database
    if engine is None:
        raise HTTPException(503, "Database is not configured")
    return engine


def create_app(
    database: str | URL | None = None, model_dir: Path | None = None, cache_url: str | None = None
) -> FastAPI:
    database = database or os.getenv(DATABASE_ENV)
    cache_url = cache_url or os.getenv(CACHE_ENV)
    model_dir = (
        model_dir if model_dir is not None else Path(os.getenv(MODEL_ENV, DEFAULT_MODEL_DIR))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.database = connect_database(database) if database else None
        try:
            app.state.predictor = load_predictor(model_dir)
        except (OSError, ValueError) as error:
            logger.warning("Model bundle unavailable: %s", error)
            app.state.predictor = None
        app.state.cache = None
        if cache_url:
            try:
                app.state.cache = PredictionCache(cache_url, metrics.cache)
            except ValueError:
                logger.warning(
                    "Redis cache URL is invalid. Predictions will use the model directly."
                )
        try:
            yield
        finally:
            if app.state.database is not None:
                app.state.database.dispose()
            if app.state.cache is not None:
                app.state.cache.close()

    app = FastAPI(title="FaultWatch", version="1", lifespan=lifespan)
    metrics = Metrics()
    app.add_middleware(RequestMetrics, metrics=metrics)

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(
            generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError):
        details = [{key: item[key] for key in ("loc", "type", "msg")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_request: Request, error: SQLAlchemyError):
        logger.warning("Database operation failed", exc_info=error)
        return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(Path(__file__).with_name("dashboard.html"), media_type="text/html")

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readiness(request: Request, response: Response) -> dict:
        db_ready = False
        if request.app.state.database is not None:
            try:
                with _database(request).connect() as connection:
                    connection.execute(select(telemetry.c.engine_id).limit(1))
                db_ready = True
            except SQLAlchemyError:
                pass
        model_ready = request.app.state.predictor is not None
        ready = db_ready and model_ready
        response.status_code = 200 if ready else 503
        return {
            "status": "ready" if ready else "unavailable",
            "database": db_ready,
            "model": model_ready,
        }

    @app.post("/v1/telemetry", status_code=201, responses={200: {"model": IngestResult}})
    def ingest(event: TelemetryEvent, request: Request, response: Response) -> IngestResult:
        with _database(request).begin() as connection:
            outcome = store_reading(connection, **event.model_dump())
        metrics.telemetry.labels(outcome=outcome).inc()
        if outcome == "conflict":
            raise HTTPException(409, "This engine and cycle already have different readings")
        if outcome == "duplicate":
            response.status_code = 200
        return IngestResult(engine_id=event.engine_id, cycle=event.cycle, status=outcome)

    @app.get("/v1/engines")
    def engines(request: Request) -> list[EngineSummary]:
        statement = (
            select(telemetry.c.engine_id, func.max(telemetry.c.cycle).label("latest_cycle"))
            .group_by(telemetry.c.engine_id)
            .order_by(telemetry.c.engine_id)
        )
        with _database(request).connect() as connection:
            return [
                EngineSummary.model_validate(row)
                for row in connection.execute(statement).mappings()
            ]

    @app.get("/v1/engines/{engine_id}/history")
    def history(
        engine_id: EnginePath, request: Request, limit: Annotated[int, Query(ge=1, le=500)] = 100
    ) -> list[StoredReading]:
        with _database(request).connect() as connection:
            rows = recent_readings(connection, engine_id, limit)
        if not rows:
            raise HTTPException(404, "Engine has no telemetry")
        return [StoredReading.model_validate(row) for row in rows]

    @app.get("/v1/engines/{engine_id}/prediction")
    def prediction(engine_id: EnginePath, request: Request) -> Prediction:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, "Model unavailable")
        with _database(request).connect() as connection:
            rows = recent_readings(connection, engine_id, WINDOW_SIZE)
        if not rows:
            raise HTTPException(404, "Engine has no telemetry")

        last_cycle = rows[-1]["cycle"]
        consecutive = 0
        for offset, row in enumerate(reversed(rows)):
            if row["cycle"] != last_cycle - offset:
                break
            consecutive += 1
        result = Prediction(
            engine_id=engine_id,
            cycle=last_cycle,
            status="insufficient_history",
            available_cycles=consecutive,
            model_version=predictor.model_version,
        )
        if consecutive < WINDOW_SIZE:
            return result

        def score() -> float:
            frame = pd.DataFrame(
                [(engine_id, row["cycle"], *row["settings"], *row["sensors"]) for row in rows],
                columns=TELEMETRY_COLUMNS,
            )
            with metrics.inference_duration.time():
                return float(predictor.predict_latest(frame).iloc[0])

        cache = request.app.state.cache
        try:
            if cache is None:
                result.predicted_rul_cycles = score()
            else:
                key = prediction_key(predictor.model_version, predictor.artifact_sha256, rows)
                result.predicted_rul_cycles = cache.get_or_score(key, score)
        except ValueError as error:
            raise HTTPException(422, "Stored telemetry cannot be scored by this model") from error
        result.status = "ready"
        return result

    return app
