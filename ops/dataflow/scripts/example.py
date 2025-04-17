from avro.io import BinaryEncoder, DatumWriter
import avro.schema as schema
import io
import json
from google.api_core.exceptions import NotFound
from google.cloud.pubsub import PublisherClient
from google.pubsub_v1.types import Encoding

PROJECT_ID = "machine-learning-456412"  # @param {type: "string"}
TOPIC_ID = "dataflow_input"  # @param {type: "string"}
AVSC_FILE = "../schema/sample.avsc"  # @param {type: "string"}

publisher_client = PublisherClient()
topic_path = publisher_client.topic_path(PROJECT_ID, TOPIC_ID)

with open(AVSC_FILE, "rb") as file:
    avro_schema = schema.parse(file.read())
writer = DatumWriter(avro_schema)

def publish_message_to_pubsub(record: dict[str, str]) -> None:
    with io.BytesIO() as bout:
        try:
            topic = publisher_client.get_topic(request={"topic": topic_path})
            encoding = topic.schema_settings.encoding

            if encoding == Encoding.BINARY:
                encoder = BinaryEncoder(bout)
                writer.write(record, encoder)
                data = bout.getvalue()
                print(f"Preparing a binary-encoded message:\n{data.decode()}")
            elif encoding == Encoding.JSON:
                data_str = json.dumps(record)
                print(f"Preparing a JSON-encoded message:\n{data_str}")
                data = data_str.encode("utf-8")
            else:
                print(f"No encoding specified in {topic_path}. Abort.")
                exit(0)

            future = publisher_client.publish(topic_path, data)
            print(f"Published message ID: {future.result()}")
        except NotFound:
            print(f"{TOPIC_ID} not found.")

record = {"name": "Alaska", "post_abbr": "AK"}
publish_message_to_pubsub(record)

import uuid
import random

for _ in range(100):
    record = {
        "name": str(uuid.uuid4())[:5],
        "post_abbr": random.choice(["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA"]),
    }
    publish_message_to_pubsub(record)