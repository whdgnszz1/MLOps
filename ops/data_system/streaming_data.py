from google.cloud import bigquery
import uuid
import time
import random
import string


def generate_random_text(length: int = 10) -> str:
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


client = bigquery.Client.from_service_account_json("../config/service-account-key.json")
table_id = "machine-learning-456412.introduction.streaming"

try:
    client.get_table(table_id)
    print("Table {} already exists.".format(table_id))
except:
    schema = [
        bigquery.SchemaField(name="log_id", field_type="STRING"),
        bigquery.SchemaField(name="text", field_type="STRING"),
        bigquery.SchemaField(name="date", field_type="INTEGER"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)
    print(
        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
    )


def insert_new_line() -> None:
    rows_to_insert = [
        {
            "log_id": str(uuid.uuid4()),
            "text": generate_random_text(50),
            "date": int(time.time()),
        },
    ]

    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))


insert_new_line()
