import mlrun
from kfp import dsl


@dsl.pipeline(name="GitOps Training Pipeline", description="Train a model")
def pipeline(
    existing_model_path: str = "None",
    force_deploy: bool = False,
    source_url: str = "https://s3.wasabisys.com/iguazio/data/model-monitoring/iris_dataset.csv",
    label_column: str = "label",
):
    # Get our project object
    project = mlrun.get_current_project()

    # Ingest data
    ingest_fn = project.get_function("get-data")
    ingest = project.run_function(
        ingest_fn,
        inputs={"source_url": source_url},
        outputs=["cleaned_data"],
    )

    # Analyze data
    describe_fn = project.get_function("describe")
    project.run_function(
        describe_fn, inputs={"table": source_url}, params={"label_column": label_column}
    )

    # Train model
    train_fn = project.get_function("train")
    train = project.run_function(
        train_fn,
        inputs={"dataset": ingest.outputs["cleaned_data"]},
        params={"label_column": label_column, "test_size": 0.10},
        outputs=["model", "test_set"],
    )

    # Evaluate model and optionally trigger deployment pipeline
    test_fn = project.get_function("test")
    project.run_function(
        test_fn,
        inputs={"test_set": train.outputs["test_set"]},
        params={
            "label_column": label_column,
            "new_model_path": train.outputs["model"],
            "existing_model_path": existing_model_path,
            "comparison_metric": "accuracy",
            "post_github": True,
            "force_deploy": force_deploy,
        },
    )
