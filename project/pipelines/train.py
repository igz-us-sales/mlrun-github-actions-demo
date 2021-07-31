from kfp import dsl
from mlrun import mount_v3io, NewTask

funcs = {}
this_project = None
DATASET = 'iris_dataset'
LABELS  = "label"

# init functions is used to configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())
        f.set_env("GITHUB_TOKEN", secrets.get("GITHUB_TOKEN"))

@dsl.pipeline(
    name="Demo training pipeline",
    description="Shows how to use mlrun."
)
def kfpipeline(
    existing_model_path:str="None"
):
    
    # run the ingestion function with the new image and params
    ingest = funcs['gen-iris'].as_step(
        name="get-data",
        handler='iris_generator',
        params={'format': 'pq'},
        outputs=[DATASET])

    # analyze our dataset
    describe = funcs["describe"].as_step(
        name="summary",
        params={"label_column": LABELS},
        inputs={"table": ingest.outputs[DATASET]})
    
    # train with hyper-paremeters
    train = funcs["train"].as_step(
        name="train",
        params={"sample"          : -1,
                "label_column"    : LABELS,
                "test_size"       : 0.10},
        hyperparams={'model_pkg_class': ["sklearn.ensemble.RandomForestClassifier",
                                         "sklearn.linear_model.LogisticRegression",
                                         "sklearn.ensemble.AdaBoostClassifier"]},
        selector='max.accuracy',
        inputs={"dataset"         : ingest.outputs[DATASET]},
        labels={"commit": this_project.params.get('commit', '')},
        outputs=['model', 'test_set'])

    # test and visualize our model
    test = funcs["test"].as_step(
        name="test",
        handler="test_classifier",
        params={"label_column": LABELS,
                "new_model_path" : train.outputs['model'],
                "existing_model_path" : existing_model_path,
                "comparison_metric": "accuracy",
                "post_github" : True},
        inputs={"test_set"    : train.outputs['test_set']})

#     # deploy our model as a serverless function
#     deploy = funcs["serving"].deploy_step(models={f"{DATASET}_v1": train.outputs['model']},
#                                           tag=this_project.params.get('commit', 'v1'))

#     # test out new model server (via REST API calls)
#     tester = funcs["live_tester"].as_step(name='model-tester',
#         params={'addr': deploy.outputs['endpoint'], 'model': f"{DATASET}_v1"},
#         inputs={'table': train.outputs['test_set']})
