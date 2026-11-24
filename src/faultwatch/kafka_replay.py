"""Replay FD001 through Kafka and commit offsets after PostgreSQL stores the row."""

import argparse
import json
from pathlib import Path
from time import monotonic

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from faultwatch.api import TelemetryEvent
from faultwatch.db import connect_database, database_url, store_reading
from faultwatch.replay import replay_events

TOPIC = "faultwatch.telemetry.v1"


def publish(broker: str, data: Path, engines: list[int] | None, topic: str) -> int:
    events = list(replay_events(data, engines))
    config = {"bootstrap.servers": broker}
    admin = AdminClient(config)
    try:
        admin.create_topics([NewTopic(topic, num_partitions=3, replication_factor=1)])[
            topic
        ].result(15)
    except KafkaException as error:
        if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
            raise
    producer = Producer({**config, "enable.idempotence": True, "message.timeout.ms": 10000})
    failures = []

    def delivery(error, _message):
        if error:
            failures.append(error)

    for event in events:
        producer.produce(
            topic, key=str(event["engine_id"]), value=json.dumps(event), on_delivery=delivery
        )
    if producer.flush(15) or failures:
        raise RuntimeError("Kafka did not acknowledge every telemetry event")
    return len(events)


def consume(
    broker: str, database, topic: str, group: str, limit: int, idle_timeout: float = 15
) -> dict:
    if limit < 1:
        raise ValueError("--limit must be positive")
    engine = connect_database(database)
    consumer = Consumer(
        {
            "bootstrap.servers": broker,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
        }
    )
    counts = {"created": 0, "duplicate": 0}
    engine_partitions = {}
    last_activity = monotonic()
    try:
        consumer.subscribe([topic])
        for _ in range(limit):
            message = consumer.poll(0.5)
            while message is None:
                if monotonic() - last_activity >= idle_timeout:
                    return {"counts": counts, "engine_partitions": engine_partitions}
                message = consumer.poll(0.5)
            if message.error():
                raise KafkaException(message.error())
            event = TelemetryEvent.model_validate_json(message.value())
            if message.key() != str(event.engine_id).encode():
                raise ValueError("Kafka key does not match telemetry engine_id")
            with engine.begin() as connection:
                status = store_reading(connection, **event.model_dump())
                if status == "conflict":
                    raise ValueError(
                        f"Conflicting telemetry at engine {event.engine_id}, cycle {event.cycle}"
                    )
            offsets = consumer.commit(message=message, asynchronous=False)
            for partition in offsets:
                if partition.error:
                    raise KafkaException(partition.error)
            counts[status] += 1
            engine_partitions[event.engine_id] = message.partition()
            last_activity = monotonic()
        return {"counts": counts, "engine_partitions": engine_partitions}
    finally:
        consumer.close()
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default="127.0.0.1:29092")
    parser.add_argument("--topic", default=TOPIC)
    commands = parser.add_subparsers(dest="command", required=True)
    publisher = commands.add_parser("publish")
    publisher.add_argument("--data", type=Path, default=Path("data/fd001"))
    publisher.add_argument("--engine", type=int, action="append")
    worker = commands.add_parser("consume")
    worker.add_argument("--group", default="faultwatch-postgres-v1")
    worker.add_argument("--limit", type=int, default=206)
    args = parser.parse_args()
    if args.command == "publish":
        result = {"published": publish(args.broker, args.data, args.engine, args.topic)}
    else:
        result = consume(args.broker, database_url(), args.topic, args.group, args.limit)
    print(json.dumps(result, indent=2))
