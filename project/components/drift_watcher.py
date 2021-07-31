import os
import json
from github import Github
from mlrun import get_run_db

def format_issue(body, model_endpoint):
    issue = "/driftdetected\n"
    issue += f"Model: {model_endpoint.spec.model}\n"
    issue += f"Model Path: {model_endpoint.spec.model_uri}\n"
    issue += f"Model Endpoint: {body['endpoint_id']}\n"
    issue += f"Drift Status: {body['drift_status']}\n"
    issue += f"Drift Measure: {body['drift_measure']}\n"
    return issue
    
def create_issue(body, model_endpoint):
    issue = format_issue(body, model_endpoint)
    
    g = Github(login_or_token=os.getenv("GITHUB_TOKEN"))
    repo = g.get_organization("igz-us-sales").get_repo("mlrun-github-actions-demo")
    repo.create_issue(f"Drift Detected - {body['drift_status']}", body=issue, assignee="nschenone")

def init_context(context):
    context.db = get_run_db()
    context.project = os.getenv("MLRUN_DEFAULT_PROJECT")
    
def handler(context, event):
    body = json.loads(event.body)
    model_endpoint = context.db.get_endpoint(project=context.project, endpoint_id=body["endpoint_id"])
    create_issue(body, model_endpoint)