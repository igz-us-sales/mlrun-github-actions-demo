import mlrun
import nuclio
from kfp import dsl


@dsl.pipeline(name="GitOps Deployment Pipeline", description="Deploy a model")
def pipeline(model_path: str = "None", label_column: str = "label"):
    # Get our project object
    project = mlrun.get_current_project()

    # Deploy model to endpoint
    serving_fn = project.get_function("serving")
    serving_fn.set_tracking()
    deploy = project.deploy_function(
        serving_fn, models=[{"key": "model", "model_path": model_path}]
    )

    # Deploy a live traffic simulator
    simulate_traffic_fn = project.get_function("simulate-traffic")
    simulate_traffic_fn.apply(mlrun.mount_v3io())
    simulate_traffic_fn.add_trigger(name="cron", spec=nuclio.triggers.CronTrigger(interval="1m"))
    project.deploy_function(
        simulate_traffic_fn,
        env={
            "model_uri": model_path,
            "dataset_name": "test_set",
            "label_column": label_column,
            "addr": deploy.outputs["endpoint"],
        },
    )

    # Deploy drift watcher
    drift_watcher_fn = project.get_function("drift-watcher")
    drift_watcher_fn.add_v3io_stream_trigger(
        name="stream",
        stream_path=f"users/pipelines/{project.name}/model-endpoints/log_stream",
        seek_to="latest",
    )
    project.deploy_function(drift_watcher_fn).after(deploy)
