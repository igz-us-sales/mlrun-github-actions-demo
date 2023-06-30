import json
import os

import mlrun
from github import Github


def format_issue(body, model_endpoint, model_uri_with_tag):
    issue_title = f"Drift Detected - {body['drift_status']}"
    issue_body = "/driftdetected\n"
    issue_body += f"Model: {model_endpoint.spec.model}\n"
    issue_body += f"Model Path: {model_uri_with_tag}\n"
    issue_body += f"Model Endpoint: {body['endpoint_id']}\n"
    issue_body += f"Drift Status: {body['drift_status']}\n"
    issue_body += f"Drift Measure: {body['drift_measure']}\n"
    return issue_title, issue_body
    

def load_model_uri_with_tag(project, model_endpoint):
    model_uri = model_endpoint.spec.model_uri
    model_tag = model_uri.split(":")[-1]
    return project.list_models(tag=model_tag)[0].uri


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
    
    # Parse model endpoint from payload
    model_endpoint = context.db.get_model_endpoint(
        project=context.project, endpoint_id=body["endpoint_id"]
    )
    
    # Load model uri and tag from project and endpoint
    project = context.db.get_project(name=context.project)
    model_uri_with_tag = load_model_uri_with_tag(project, model_endpoint)
    
    # Format github issue
    issue_title, issue_body = format_issue(body, model_endpoint, model_uri_with_tag)
    
    # Authenticate repo
    g = Github(login_or_token=mlrun.get_secret_or_env("MY_GITHUB_TOKEN"))
    repo = g.get_organization("igz-us-sales").get_repo("mlrun-github-actions-demo")
    
    # Create GitHub issue
    repo.create_issue(issue_title, body=issue_body, assignee="nschenone")

    # Trigger re-training
    trigger_retrain(repo, model_uri_with_tag)