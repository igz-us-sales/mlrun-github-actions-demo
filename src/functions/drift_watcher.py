import json
import os

import mlrun
from github import Github


def format_issue(body, model_endpoint):
    issue = "/driftdetected\n"
    issue += f"Model: {model_endpoint.spec.model}\n"
    issue += f"Model Path: {model_endpoint.spec.model_uri}\n"
    issue += f"Model Endpoint: {body['endpoint_id']}\n"
    issue += f"Drift Status: {body['drift_status']}\n"
    issue += f"Drift Measure: {body['drift_measure']}\n"
    return issue


def create_issue(body, model_endpoint):
    # Format GitHub issue with drift notification results
    issue = format_issue(body, model_endpoint)

    # Authenticate repo
    g = Github(login_or_token=mlrun.get_secret_or_env("MY_GITHUB_TOKEN"))
    repo = g.get_organization("igz-us-sales").get_repo("mlrun-github-actions-demo")

    # Create issue
    repo.create_issue(f"Drift Detected - {body['drift_status']}", body=issue, assignee="nschenone")

    # Trigger re-training
    trigger_retrain(repo, model_endpoint.spec.model_uri)


def trigger_retrain(repo, existing_model_path):
    retrain_workflow = [x for x in repo.get_workflows() if x.name == "training-workflow"][0]
    retrain_workflow.create_dispatch(
        ref="master", inputs={"existing_model_path": existing_model_path}
    )


def init_context(context):
    context.db = mlrun.get_run_db()
    context.project = os.getenv("MLRUN_DEFAULT_PROJECT")


def handler(context, event):
    body = json.loads(event.body)
    model_endpoint = context.db.get_model_endpoint(
        project=context.project, endpoint_id=body["endpoint_id"]
    )
    create_issue(body, model_endpoint)
