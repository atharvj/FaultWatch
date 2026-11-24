"""Store telemetry with one row per engine and cycle."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION

revision = "20261123_telemetry"
down_revision = None


def upgrade() -> None:
    op.create_table(
        "telemetry",
        sa.Column("engine_id", sa.Integer, primary_key=True),
        sa.Column("cycle", sa.Integer, primary_key=True),
        sa.Column("settings", ARRAY(DOUBLE_PRECISION), nullable=False),
        sa.Column("sensors", ARRAY(DOUBLE_PRECISION), nullable=False),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("engine_id > 0 AND cycle > 0", name="telemetry_positive_keys"),
        sa.CheckConstraint(
            "array_ndims(settings) = 1 AND cardinality(settings) = 3",
            name="telemetry_settings_shape",
        ),
        sa.CheckConstraint(
            "array_ndims(sensors) = 1 AND cardinality(sensors) = 21",
            name="telemetry_sensors_shape",
        ),
    )


def downgrade() -> None:
    op.drop_table("telemetry")
