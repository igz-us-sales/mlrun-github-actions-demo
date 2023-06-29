# Generated by nuclio.export.NuclioExporter

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

from cloudpickle import dumps
import pandas as pd
from typing import List
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.mlutils.data import get_sample, get_splits
from mlrun.mlutils.models import gen_sklearn_model, eval_model_v2
from mlrun.utils.helpers import create_class
from mlrun.artifacts.model import ModelArtifact
import mlrun

@mlrun.handler(outputs=["model", "test_set"])
def train_model(
    context: MLClientCtx,
    model_pkg_class: str,
    dataset: DataItem,
    label_column: str = "labels",
    encode_cols: List[str] = [],
    sample: int = -1,
    test_size: float = 0.30,
    train_val_split: float = 0.70,
    test_set_key: str = "test_set",
    model_evaluator=None,
    models_dest: str = "",
    plots_dest: str = "plots",
    file_ext: str = "parquet",
    model_pkg_file: str = "",
    random_state: int = 1,
) -> None:
    """train a classifier

    An optional cutom model evaluator can be supplied that should have the signature:
    `my_custom_evaluator(context, xvalid, yvalid, model)` and return a dictionary of
    scalar "results", a "plots" keys with a list of PlotArtifacts, and
    and "tables" key containing a returned list of TableArtifacts.

    :param context:           the function context
    :param model_pkg_class:   the model to train, e.g, "sklearn.neural_networks.MLPClassifier",
                              or json model config
    :param dataset:           ("data") name of raw data file
    :param label_column:      ground-truth (y) labels
    :param encode_cols:       dictionary of names and prefixes for columns that are
                              to hot be encoded.
    :param sample:            Selects the first n rows, or select a sample
                              starting from the first. If negative <-1, select
                              a random sample
    :param test_size:         (0.05) test set size
    :param train_val_split:   (0.75) Once the test set has been removed the
                              training set gets this proportion.
    :param test_set_key:      key of held out data in artifact store
    :param model_evaluator:   (None) a custom model evaluator can be specified
    :param models_dest:       ("") models subfolder on artifact path
    :param plots_dest:        plot subfolder on artifact path
    :param file_ext:          ("parquet") format for test_set_key hold out data
    :param random_state:      (1) sklearn rng seed

    """
    models_dest = models_dest or "model"

    raw, labels, header = get_sample(dataset, sample, label_column)

    if encode_cols:
        raw = pd.get_dummies(
            raw,
            columns=list(encode_cols.keys()),
            prefix=list(encode_cols.values()),
            drop_first=True,
        )

    (xtrain, ytrain), (xvalid, yvalid), (xtest, ytest) = get_splits(
        raw, labels, 3, test_size, 1 - train_val_split, random_state
    )

    test_set = pd.concat([xtest, ytest.to_frame()], axis=1)

    context.log_dataset(
        test_set_key,
        df=test_set,
        format=file_ext,
        index=False,
        labels={"data-type": "held-out"},
        artifact_path=context.artifact_subpath("data"),
    )

    model_config = gen_sklearn_model(model_pkg_class, context.parameters.items())

    model_config["FIT"].update({"X": xtrain, "y": ytrain.values})

    ClassifierClass = create_class(model_config["META"]["class"])

    model = ClassifierClass(**model_config["CLASS"])

    model.fit(**model_config["FIT"])

    artifact_path = context.artifact_subpath(models_dest)
    plots_path = context.artifact_subpath(models_dest, plots_dest)
    if model_evaluator:
        eval_metrics = model_evaluator(
            context, xvalid, yvalid, model, plots_artifact_path=plots_path
        )
    else:
        # eval_metrics = eval_model_v2(
        #     context, xvalid, yvalid, model, plots_artifact_path=plots_path
        # )
        eval_metrics = 0

    kwargs = {}
    if "algorithm" in ModelArtifact._dict_fields:
        kwargs["training_set"] = xtrain
#         kwargs["label_column"] = label_column
        split = model_pkg_class.rsplit(".", 1)
        if split and len(split) == 2:
            kwargs["algorithm"] = split[1]

        if dataset.meta and dataset.meta.kind == "FeatureVector":
            kwargs["feature_vector"] = dataset.meta.uri

    context.set_label("class", model_pkg_class)
    context.log_model(
        "model",
        body=dumps(model),
        artifact_path=artifact_path,
        extra_data=eval_metrics,
        model_file="model.pkl",
        metrics=context.results,
        labels={"class": model_pkg_class},
        framework="sklearn",
        **kwargs
    )

    return model, test_set