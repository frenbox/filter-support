import os
import time
import json

import io
import fastavro

from confluent_kafka import Consumer
from dotenv import load_dotenv
from superphot_boom import run_superphot
def read_avro(msg):
    """
    Reads an Avro record from a Kafka message.

    Args:
        msg: The message object containing the Avro data.

    Returns:
        The first record found in the Avro message, or None if no records are found.
    """

    bytes_io = io.BytesIO(msg.value())  # Get the message value as bytes
    bytes_io.seek(0)
    for record in fastavro.reader(bytes_io):
        return record  # Return the first record found
    return None  # Return None if no records are found or if an error occurs


consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': f'umn_boom_kafka_consumer_group_{int(time.time())}',
    'auto.offset.reset': 'earliest',
    "enable.auto.commit": False,  # Disable auto-commit of offsets
    "session.timeout.ms": 6000,  # Session timeout for the consumer
    "max.poll.interval.ms": 300000,  # Maximum time between polls
    "security.protocol": "PLAINTEXT",  # Use PLAINTEXT if no authentication
})
consumer.subscribe(['ZTF_alerts_results'])

thumbnail_types = [
    ("cutoutScience", "new"),
    ("cutoutTemplate", "ref"),
    ("cutoutDifference", "sub"),
]

print("Subscribed to topic: ZTF_alerts_results")

def consume():
    print("Listening for messages...")
    alerts = 0
    try:
        while True:
            msg = consumer.poll(timeout=10.0)
            if msg is None:
                print(f"No {'more ' if alerts else ''}messages available, exiting")
                break
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            record = read_avro(msg)

            # Remove cutouts to improve readability, you can remove this block to keep them
            for cutout_type, _ in thumbnail_types:
                del record[cutout_type]

            # Save the first alert to a JSON file for inspection of its structure
            if alerts == 0:
                with open("first_alert.json", "w") as f:
                    json.dump(record, f, indent=2)

            ztf_id = record["objectId"]
            for passed_filter in record["filters"]:
                if passed_filter["filter_name"] == "fast_transient_ztf":
                    run_superphot(ztf_id)
                    alerts += 1

    except KeyboardInterrupt:
        pass
    finally:
        print(f"Processed {alerts} messages")
        consumer.close()

    consumer.close()

consume()
