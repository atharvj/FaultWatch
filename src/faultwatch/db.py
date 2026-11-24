"""PostgreSQL storage for FD001 telemetry."""

import os
from typing import Literal

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Integer,
    MetaData,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, insert
from sqlalchemy.engine import Connection, Engine, URL

DATABASE_ENV = "FAULTWATCH_DATABASE_URL"

metadata = MetaData()
telemetry = Table(
    "telemetry",
    metadata,
    Column("engine_id", Integer, primary_key=True),
    Column("cycle", Integer, primary_key=True),
    Column("settings", ARRAY(DOUBLE_PRECISION), nullable=False),
    Column("sensors", ARRAY(DOUBLE_PRECISION), nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("engine_id > 0 AND cycle > 0", name="telemetry_positive_keys"),
    CheckConstraint(
        "array_ndims(settings) = 1 AND cardinality(settings) = 3",
        name="telemetry_settings_shape",
    ),
    CheckConstraint(
        "array_ndims(sensors) = 1 AND cardinality(sensors) = 21",
        name="telemetry_sensors_shape",
    ),
)


def database_url() -> str:
    url = os.environ.get(DATABASE_ENV)
    if not url:
        raise ValueError(f"Set {DATABASE_ENV} to a PostgreSQL connection URL")
    return url


def connect_database(url: str | URL) -> Engine:
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3})


def store_reading(
    connection: Connection, engine_id: int, cycle: int, settings: list[float], sensors: list[float]
) -> Literal["created", "duplicate", "conflict"]:
    statement = (
        insert(telemetry)
        .values(engine_id=engine_id, cycle=cycle, settings=settings, sensors=sensors)
        .on_conflict_do_nothing(index_elements=[telemetry.c.engine_id, telemetry.c.cycle])
        .returning(telemetry.c.engine_id)
    )
    if connection.execute(statement).scalar_one_or_none() is not None:
        return "created"

    existing = connection.execute(
        select(telemetry.c.settings, telemetry.c.sensors).where(
            telemetry.c.engine_id == engine_id, telemetry.c.cycle == cycle
        )
    ).one()
    if existing.settings == settings and existing.sensors == sensors:
        return "duplicate"
    return "conflict"


def recent_readings(connection: Connection, engine_id: int, limit: int):
    statement = (
        select(telemetry)
        .where(telemetry.c.engine_id == engine_id)
        .order_by(telemetry.c.cycle.desc())
        .limit(limit)
    )
    return list(reversed(connection.execute(statement).mappings().all()))
