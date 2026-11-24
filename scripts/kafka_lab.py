"""Verify persisted offsets, engine ordering, and idempotent Kafka replay."""

import json
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from confluent_kafka.admin import AdminClient
from sqlalchemy import make_url, select
from sqlalchemy.schema import CreateSchema, DropSchema

from faultwatch.db import connect_database, database_url, telemetry
from faultwatch.kafka_replay import consume, publish

BROKER = "127.0.0.1:29092"


def main():
    namespace = f"faultwatch_test_{uuid4().hex}"
    topic = f"{namespace}.v1"
    url = make_url(database_url())
    engine = connect_database(url)
    isolated = url.update_query_dict({"options": f"-csearch_path={namespace}"})
    config = Config("alembic.ini")
    config.attributes["database_url"] = isolated
    with engine.begin() as connection:
        connection.execute(CreateSchema(namespace))
    try:
        command.upgrade(config, "head")
        total = publish(BROKER, Path("data/fd001"), [1, 2], topic)
        first = consume(BROKER, isolated, topic, namespace, 25)
        second = consume(BROKER, isolated, topic, namespace, total - 25)
        assert first["counts"] == {"created": 25, "duplicate": 0}
        assert second["counts"] == {"created": total - 25, "duplicate": 0}
        for engine_id in first["engine_partitions"].keys() & second["engine_partitions"].keys():
            assert first["engine_partitions"][engine_id] == second["engine_partitions"][engine_id]

        publish(BROKER, Path("data/fd001"), [1, 2], topic)
        replay = consume(BROKER, isolated, topic, namespace, total)
        assert replay["counts"] == {"created": 0, "duplicate": total}

        database = connect_database(isolated)
        try:
            with database.connect() as connection:
                rows = connection.execute(
                    select(telemetry.c.engine_id, telemetry.c.cycle).order_by(
                        telemetry.c.received_at, telemetry.c.cycle
                    )
                ).all()
        finally:
            database.dispose()
        by_engine = {
            engine_id: [cycle for key, cycle in rows if key == engine_id] for engine_id in (1, 2)
        }
        assert len(rows) == total
        assert all(cycles == list(range(1, len(cycles) + 1)) for cycles in by_engine.values())
        print(
            json.dumps(
                {
                    "published": total,
                    "first_consumer": first,
                    "restarted_consumer": second,
                    "second_publication": replay["counts"],
                    "stored_cycles_by_engine": {
                        engine_id: len(cycles) for engine_id, cycles in by_engine.items()
                    },
                    "engine_order_verified": True,
                },
                indent=2,
            )
        )
    finally:
        with engine.begin() as connection:
            connection.execute(DropSchema(namespace, cascade=True))
        engine.dispose()
        admin = AdminClient({"bootstrap.servers": BROKER})
        if topic in admin.list_topics(timeout=5).topics:
            admin.delete_topics([topic])[topic].result(10)


if __name__ == "__main__":
    main()
