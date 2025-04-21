from datasets import load_dataset
from google.oauth2 import service_account
import pandas_gbq
import pandas as pd

credentials = service_account.Credentials.from_service_account_file(
    "../config/service-account-key.json",
)

dataset = load_dataset("wikipedia", language="en", date="20220301")

df = pd.DataFrame(dataset["train"][:100000])
pandas_gbq.to_gbq(
    df,
    "introduction.wikipedia",
    project_id="machine-learning-456412",
    credentials=credentials,
)