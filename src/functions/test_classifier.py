import warnings

import mlrun
import numpy as np
import pandas as pd
import sklearn
from cloudpickle import load
from github import Github
from mlrun.artifacts import get_model, update_model

warnings.filterwarnings("ignore")


def eval_model(context, xtest, ytest, model):
    ypred = model.predict(xtest)
    metrics = {
        "accuracy": float(sklearn.metrics.accuracy_score(ytest, ypred)),
        "test-error": np.sum(ytest != ypred) / ytest.shape[0],
        "f1": float(sklearn.metrics.f1_score(ytest, ypred, average="macro")),
        "precision": float(sklearn.metrics.precision_score(ytest, ypred, average="macro")),
        "recall": float(sklearn.metrics.recall_score(ytest, ypred, average="macro")),
    }
    return ypred, metrics


def format_issue(model_name, models):
    body = f"  - {model_name}: {models[model_name]['model_path']}\n"
    body += f"    - Accuracy: {models[model_name]['metrics']['accuracy']}\n"
    body += f"    - F1: {models[model_name]['metrics']['f1']}\n"
    body += f"    - Precision: {models[model_name]['metrics']['precision']}\n"
    body += f"    - Recall: {models[model_name]['metrics']['recall']}\n"
    return body


def create_issue(context, models, comparison_metric):
    deploy_new_model = True

    # Format GitHub issue with train run results
    body = format_issue("new_model", models)
    if "existing_model" in models:
        body += format_issue("existing_model", models)
        deploy_new_model = (
            models["new_model"]["metrics"][comparison_metric]
            > models["existing_model"]["metrics"][comparison_metric]
        )

    # Authenticate repo
    g = Github(login_or_token=mlrun.get_secret_or_env("MY_GITHUB_TOKEN"))
    repo = g.get_organization("igz-us-sales").get_repo("mlrun-github-actions-demo")

    # Create issue
    repo.create_issue(f"Train Results - Run {context.uid}", body=body, assignee="xsqian")

    # Trigger deployment based on model metrics
    if context.get_param("force_deploy") or deploy_new_model:
        trigger_deployment(context, repo, models["new_model"]["model_path"])


def trigger_deployment(context, repo, model_path):
    context.logger.info("TRIGGER_DEPLOYMENT")
    deploy_workflow = [x for x in repo.get_workflows() if x.name == "deploy-workflow"][0]
    deploy_workflow.create_dispatch(ref="master", inputs={"model_path": model_path})
    context.logger.info(deploy_workflow)


def test_classifier(
    context,
    new_model_path,
    test_set,
    label_column: str,
    comparison_metric: str = "accuracy",
    post_github: bool = False,
    predictions_column: str = "yscore",
    model_update=True,
) -> None:
    # Load test data
    print(test_set)
    xtest = test_set.as_df()
    ytest = xtest.pop(label_column)

    # Build model name/path/metrics dict
    models = {}
    models["new_model"] = {"model_path": new_model_path}

    #     # Evaluate existing model if parameter is passed
    if context.get_param("existing_model_path"):
        models["existing_model"] = {"model_path": context.get_param("existing_model_path")}

    for model_name, model_config in models.items():
        # Load model
        try:
            model_file, model_obj, _ = get_model(model_config["model_path"], suffix=".pkl")
            model_obj = load(open(model_file, "rb"))
        except Exception as a:
            raise Exception("model location likely specified")

        # Evalaute
        ypred, metrics = eval_model(context, xtest, ytest.values, model_obj)
        models[model_name]["metrics"] = metrics

        # Log metrics per model
        for metric, value in metrics.items():
            context.log_result(f"{metric}-{model_name}", value)

        # Update model artifact with metrics
        if model_obj and model_update == True:
            update_model(
                model_config["model_path"],
                metrics=metrics,
                key_prefix="validation-",
            )

        # Get test set column names
        if ypred.ndim == 1 or ypred.shape[1] == 1:
            score_names = [predictions_column]
        else:
            score_names = [f"{predictions_column}_" + str(x) for x in range(ypred.shape[1])]

        # Log test set predictions
        df = pd.concat([xtest, ytest, pd.DataFrame(ypred, columns=score_names)], axis=1)
        context.log_dataset(f"test_set_preds-{model_name}", df=df, format="parquet", index=False)

    # Create GitHub issue for run
    if post_github:
        create_issue(context, models, comparison_metric)
