import importlib

import mlrun

IMAGE_REQUIREMENTS = ["mlrun==1.3.3", "PyGithub==1.59.0"]


def assert_build():
    for module_name in IMAGE_REQUIREMENTS:
        name, version = module_name.split("==")
        module = importlib.import_module(name)
        print(module.__version__)
        assert module.__version__ == version


def create_and_set_project(
    git_source: str,
    name: str = "gitops-flow",
    default_image: str = None,
    default_base_image: str = "mlrun/mlrun:1.3.3",
    user_project: bool = False,
    env_file: str = None,
    force_build: bool = False,
):
    """
    Creating the project for this demo.
    :param git_source:              the git source of the project.
    :param name:                    project name
    :param default_image:           the default image of the project
    :param user_project:            whether to add username to the project name

    :returns: a fully prepared project for this demo.
    """

    # Set MLRun DB endpoint
    if env_file:
        mlrun.set_env_from_file(env_file=env_file)

    # Get / Create a project from the MLRun DB:
    project = mlrun.get_or_create_project(name=name, context="./", user_project=user_project)

    # Set or build the default image:
    if force_build or project.default_image is None:
        if default_image is None:
            print("Building default project image...")
            image_builder = project.set_function(
                func="src/project_setup.py",
                name="image-builder",
                handler="assert_build",
                kind="job",
                image=default_base_image
            )
            build_status = project.build_function(
                function=image_builder,
                base_image=default_base_image,
                requirements=IMAGE_REQUIREMENTS,
            )
            default_image = build_status.outputs["image"]
        project.set_default_image(default_image)

    # Export project to zip if relevant
    if ".zip" in git_source:
        project.export(git_source)

    # Set the project git source
    project.set_source(git_source, pull_at_runtime=True)

    # Set MLRun functions
    project.set_function(
        name="get-data",
        func="src/functions/get_data.py",
        kind="job",
        handler="get_data",
    )
    project.set_function(
        name="describe",
        func="hub://describe",
        kind="job",
        handler="analyze",
    )
    project.set_function(
        name="train",
        func="src/functions/train.py",
        kind="job",
        handler="train_model",
    )
    project.set_function(
        name="test",
        func="src/functions/test_classifier.py",
        kind="job",
        handler="test_classifier",
    )
    project.set_function(
        name="create-drift-stream",
        func="src/functions/create_drift_stream.py",
        kind="job",
        handler="create_drift_stream",
    )
    project.set_function(
        name="serving",
        func="hub://v2_model_server",
        kind="serving",
        image=default_base_image,
        requirements=IMAGE_REQUIREMENTS,
    )
    project.set_function(
        name="simulate-traffic",
        func="src/functions/simulate_traffic.py",
        kind="nuclio",
        image=default_base_image,
        requirements=IMAGE_REQUIREMENTS,
    )
    project.set_function(
        name="drift-watcher",
        func="src/functions/drift_watcher.py",
        kind="nuclio",
        image=default_base_image,
        requirements=IMAGE_REQUIREMENTS,
    )

    # Set MLRun workflows
    project.set_workflow(name="train", workflow_path="src/workflows/train.py")
    project.set_workflow(name="deploy", workflow_path="src/workflows/deploy.py")

    # Save and return the project:
    project.save()
    return project
