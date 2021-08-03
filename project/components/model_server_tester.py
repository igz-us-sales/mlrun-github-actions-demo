import os
import json
import random
import requests
from time import sleep
from mlrun.datastore import get_store_resource

def init_context(context):
    context.addr = os.getenv("addr") + "/v2/models/model/infer"
    context.model_path = os.getenv("model_path")
#     context.data_path = f"store://datasets/gitops-project/train_test_set@{context.model_path.split('@')[1]}"
    context.data_path = f"store://datasets/gitops-project/train_test_set:{context.model_path.split(':')[2]}"
    df = get_store_resource(context.data_path).to_dataitem().as_df().drop("label", axis=1)
    context.data = df.values.tolist()

def handler(context, event):
    data = json.dumps({'inputs': [random.choice(context.data)]})
    requests.post(url=context.addr, data=data)
    return 