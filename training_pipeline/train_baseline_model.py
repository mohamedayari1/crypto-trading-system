import json
from collections import OrderedDict
import os
from pathlib import Path
from typing import OrderedDict as OrderedDictType, Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sktime.performance_metrics.forecasting import (
    mean_squared_percentage_error,
    mean_absolute_percentage_error,
)
from sktime.utils.plotting import plot_series

import utils
# from training_pipeline.settings import OUTPUT_DIR
from data import load_dataset_from_feature_store
from models import build_baseline_model


import logging

OUTPUT_DIR = "/home/mohamed-ayari/projects/mohamed/crypto-trading-system/training_pipeline/output"

def get_logger(name: str) -> logging.Logger:
    """
    Template for getting a logger.

    Args:
        name: Name of the logger.

    Returns: Logger.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)

    return logger

logger = get_logger(__name__)

def train_baseline_model(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    training_dataset_version: Optional[int] = None,
) -> dict:
    """Train and evaluate a baseline model on the test set.

    Args:
        fh (int, optional): Forecasting horizon. Defaults to 24.
        feature_view_version (Optional[int], optional): Feature store - feature view version. Defaults to None.
        training_dataset_version (Optional[int], optional): Feature store - training dataset version. Defaults to None.

    Returns:
        dict: Dictionary containing metadata about the training experiment.
    """

    # feature_view_metadata = utils.load_json("feature_view_metadata.json")
    # if feature_view_version is None:
    #     feature_view_version = feature_view_metadata["feature_view_version"]
    # if training_dataset_version is None:
    #     training_dataset_version = feature_view_metadata["training_dataset_version"]

    y_train, y_test, X_train, X_test = load_dataset_from_feature_store(
        feature_view_version=1,
        training_dataset_version=1,
        fh=fh,
    )


   # Ensure the indices have a frequency
    y_train = y_train.copy()
    # Create temporary series to apply asfreq
    temp_series = pd.Series(range(len(y_train.index.levels[1])), index=y_train.index.levels[1])
    temp_series = temp_series.asfreq('H')
    # Set the levels back
    y_train.index = y_train.index.set_levels(temp_series.index, level='timestamp')

    y_test = y_test.copy()
    # Create temporary series to apply asfreq
    temp_series = pd.Series(range(len(y_test.index.levels[1])), index=y_test.index.levels[1])
    temp_series = temp_series.asfreq('H')
    # Set the levels back
    y_test.index = y_test.index.set_levels(temp_series.index, level='timestamp')

    # Verify the index type and frequency
    logger.info(f"Index type: {type(y_train.index.levels[1])}")
    logger.info(f"Index frequency: {y_train.index.levels[1].freq}")
    

    
    training_start_datetime = y_train.index.get_level_values("timestamp").min()
    training_end_datetime = y_train.index.get_level_values("timestamp").max()
    testing_start_datetime = y_test.index.get_level_values("timestamp").min()
    testing_end_datetime = y_test.index.get_level_values("timestamp").max()
    
    
    
    logger.info(
        f"Training baseline model on data from {training_start_datetime} to {training_end_datetime}."
    )
    logger.info(
        f"Testing baseline model on data from {testing_start_datetime} to {testing_end_datetime}."
    )

    with utils.init_wandb_run(
        name="baseline_model",
        job_type="train_baseline_model",
        group="train",
        reinit=True,
        add_timestamp_to_name=True,
    ) as run:
        run.use_artifact("split_train:latest")
        run.use_artifact("split_test:latest")

        # Train baseline model
        baseline_forecaster = build_baseline_model(seasonal_periodicity=fh)
        baseline_forecaster = train_model(baseline_forecaster, y_train, X_train, fh=fh)
        y_pred, metrics_baseline = evaluate(baseline_forecaster, y_test, X_test)
    #     slices = metrics_baseline.pop("slices")
        for k, v in metrics_baseline.items():
            logger.info(f"Baseline test {k}: {v}")
        wandb.log({"test": {"baseline": metrics_baseline}})
    #     wandb.log({"test.baseline.slices": wandb.Table(dataframe=slices)})

        # Render baseline model results
        results = OrderedDict({"y_train": y_train, "y_test": y_test, "y_pred": y_pred})
        print(results)
        # render(results, prefix="images_baseline")

    #     # Save baseline model
        save_model_path = OUTPUT_DIR + "/baseline_model.pkl"
        utils.save_model(baseline_forecaster, save_model_path)

        metadata = {
            "experiment": {
                "fh": fh,
                "feature_view_version": feature_view_version,
                "training_dataset_version": training_dataset_version,
                "training_start_datetime": training_start_datetime.isoformat(),
                "training_end_datetime": training_end_datetime.isoformat(),
                "testing_start_datetime": testing_start_datetime.isoformat(),
                "testing_end_datetime": testing_end_datetime.isoformat(),
            },
            # "results": {"test": metrics_baseline},
        }
        run.finish()

    utils.save_json(metadata, file_name="baseline_train_metadata.json")
    return metadata


def train_model(model, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int):
    """Train the forecaster on the given training set and forecast horizon."""
    fh = np.arange(fh) + 1
    model.fit(y_train, X=X_train, fh=fh)
    return model


def evaluate(
    forecaster, y_test: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, dict]:
    """Evaluate the forecaster on the test set by computing the following metrics:
        - RMSPE
        - MAPE
        - Slices: RMSPE, MAPE
    """
    y_pred = forecaster.predict(X=X_test)

    # Compute aggregated metrics
    results = dict()
    rmspe = mean_squared_percentage_error(y_test, y_pred, squared=False)
    results["RMSPE"] = rmspe
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    results["MAPE"] = mape

    # # Compute metrics per slice
    # y_test_slices = y_test.groupby(["area", "consumer_type"])
    # y_pred_slices = y_pred.groupby(["area", "consumer_type"])
    # slices = pd.DataFrame(columns=["area", "consumer_type", "RMSPE", "MAPE"])
    # for y_test_slice, y_pred_slice in zip(y_test_slices, y_pred_slices):
    #     (area_y_test, consumer_type_y_test), y_test_slice_data = y_test_slice
    #     (area_y_pred, consumer_type_y_pred), y_pred_slice_data = y_pred_slice

    #     assert (
    #         area_y_test == area_y_pred and consumer_type_y_test == consumer_type_y_pred
    #     ), "Slices are not aligned."

    #     rmspe_slice = mean_squared_percentage_error(
    #         y_test_slice_data, y_pred_slice_data, squared=False
    #     )
    #     mape_slice = mean_absolute_percentage_error(
    #         y_test_slice_data, y_pred_slice_data, symmetric=False
    #     )

    #     slice_results = pd.DataFrame(
    #         {
    #             "area": [area_y_test],
    #             "consumer_type": [consumer_type_y_test],
    #             "RMSPE": [rmspe_slice],
    #             "MAPE": [mape_slice],
    #         }
    #     )
    #     slices = pd.concat([slices, slice_results], ignore_index=True)

    # results["slices"] = slices
    return y_pred, results


def render(
    timeseries: OrderedDictType[str, pd.DataFrame],
    prefix: Optional[str] = None,
    delete_from_disk: bool = True,
):
    """Render the timeseries as a single plot per (area, consumer_type) and save them to disk and wandb."""
    grouped_timeseries = OrderedDict()
    for split, df in timeseries.items():
        df = df.reset_index(level=[0, 1])
        groups = df.groupby(["area", "consumer_type"])
        for group_name, split_group_values in groups:
            group_values = grouped_timeseries.get(group_name, {})
            grouped_timeseries[group_name] = {
                f"{split}": split_group_values["energy_consumption"],
                **group_values,
            }

    output_dir = OUTPUT_DIR / prefix if prefix else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for group_name, group_values_dict in grouped_timeseries.items():
        fig, ax = plot_series(
            *group_values_dict.values(), labels=group_values_dict.keys()
        )
        fig.suptitle(f"Area: {group_name[0]} - Consumer type: {group_name[1]}")

        # Save matplotlib image
        image_save_path = str(output_dir / f"{group_name[0]}_{group_name[1]}.png")
        plt.savefig(image_save_path)
        plt.close(fig)

        if prefix:
            wandb.log({prefix: wandb.Image(image_save_path)})
            
        else:
            wandb.log(wandb.Image(image_save_path))

        if delete_from_disk:
            os.remove(image_save_path)


if __name__ == "__main__":
    fire.Fire(train_baseline_model)
    
    
