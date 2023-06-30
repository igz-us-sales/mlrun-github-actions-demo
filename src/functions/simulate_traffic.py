import os
from random import choice, randint, uniform
from time import sleep

import pandas as pd
import requests
from mlrun.artifacts import get_model

PREDICT_ROUTE = "v2/models/model/predict"


def get_data(model_uri, dataset_name, label_column):
    *_, extra_data = get_model(model_uri)
    data_uri = extra_data[dataset_name].url.replace("v3io:///", "/v3io/")
    df = pd.read_parquet(data_uri).drop(label_column, axis=1)
    return df.to_dict(orient="split")["data"]


def init_context(context):
    context.addr = os.getenv("addr")
    context.test_data = get_data(
        model_uri=os.getenv("model_uri"),
        dataset_name=os.getenv("dataset_name"),
        label_column=os.getenv("label_column"),
    )


def handler(context, event):
    for i in range(randint(10, 30)):
        data_point = choice(context.test_data)
        resp = requests.post(url=f"{context.addr}/{PREDICT_ROUTE}", json={"inputs": [data_point]})
        print(resp.json())
        sleep(uniform(0.2, 1.7))
    return
