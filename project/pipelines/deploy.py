from kfp import dsl
from mlrun import mount_v3io
import nuclio

funcs = {}
this_project = None
DATASET = 'iris_dataset'
LABELS  = "label"

# init functions is used to configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())
        f.set_env("GITHUB_TOKEN", secrets.get("MY_GITHUB_TOKEN"))
        
    # Enable model monitoring
    functions["serving"].set_tracking()
    functions["live-tester"].add_trigger('cron', nuclio.triggers.CronTrigger(interval="1s"))
    functions["drift-watcher"].add_v3io_stream_trigger(name="stream",
                                                       stream_path=f"projects/{project.metadata.name}/model-endpoints/log_stream",
                                                       seek_to="latest")
    
@dsl.pipeline(
    name="GitOps Deployment Pipeline",
    description="Deploy a model."
)
def kfpipeline(
    model_path:str="None"
):

    # deploy our model as a serverless function
    deploy = funcs["serving"].deploy_step(models={f"model": model_path})

    # test out new model server (via REST API calls)
    tester = funcs["live-tester"].deploy_step(env={"addr" : deploy.outputs["endpoint"],"model_path" : model_path})
    
    # drift watcher to post on github
    watcher = funcs["drift-watcher"].deploy_step().after(deploy)
