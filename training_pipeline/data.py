from typing import Tuple, Union, Optional
import hopsworks
import pandas as pd
import wandb

import fire


from sktime.forecasting.model_selection import temporal_train_test_split



# from training_pipeline.settings import SETTINGS
# from training_pipeline.utils import init_wandb_run


# List of important cryptocurrencies (can be extended as needed)
IMPORTANT_CRYPTOCURRENCIES = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "LTC-USD"]

FS_API_KEY = "Pg6hKTom6MjSnbNF.VBmTgApGW1pB8bCTK0MPGoixrydLJ9V92nrOjC4wAsfGnnu8ZOC7fFR3iUJL2qQQ"
FS_PROJECT_NAME = "crypto_trading_system"

WANDB_API_KEY = "2aebcfef432ed4e4bd33b7f9532cdb0bbf86586b"
WANDB_ENTITY = "ayarim781-sup-com"
WANDB_PROJECT = "crypto_trading_system"




def init_wandb_run(
    name: str,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    add_timestamp_to_name: bool = False,
    run_id: Optional[str] = None,
    resume: Optional[str] = None,
    reinit: bool = False,
    project: str = WANDB_PROJECT,
    entity: str = WANDB_ENTITY,
):
    """Wrapper over the wandb.init function."""

    if add_timestamp_to_name:
        name = f"{name}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        id=run_id,
        reinit=reinit,
        resume=resume,
    )

    return run



def load_dataset_from_feature_store(
    feature_view_version: int = 1, 
    training_dataset_version: int = 1, 
    fh: int = 24

) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """Load features from feature store.

    Args:
        feature_view_version (int): feature store feature view version to load data from
        training_dataset_version (int): feature store training dataset version to load data from
        fh (int, optional): Forecast horizon. Defaults to 24.

    Returns:
        Train and test splits loaded from the feature store as pandas dataframes.
    """

    # project = hopsworks.login(
    #     api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    # )
    
    project = hopsworks.login(
        api_key_value=FS_API_KEY, project=FS_PROJECT_NAME
    )
    fs = project.get_feature_store()

    with init_wandb_run(
        name="load_training_data", job_type="load_feature_view", group="dataset"
    ) as run:
        feature_view = fs.get_feature_view(
            name="ohlc_data_prototype_view", version=feature_view_version
        )
        data, _ = feature_view.get_training_data(
            training_dataset_version=training_dataset_version
        )

        fv_metadata = feature_view.to_dict()
        fv_metadata["query"] = fv_metadata["query"].to_string()
        fv_metadata["features"] = [f.name for f in fv_metadata["features"]]
        fv_metadata["link"] = feature_view._feature_view_engine._get_feature_view_url(
            feature_view
        )
        fv_metadata["feature_view_version"] = feature_view_version
        fv_metadata["training_dataset_version"] = training_dataset_version

        raw_data_at = wandb.Artifact(
            name="ohlc_data_prototype_view",
            type="feature_view",
            metadata=fv_metadata,
        )
        run.log_artifact(raw_data_at)

        run.finish()

    # with init_wandb_run(
    #     name="train_test_split", job_type="prepare_dataset", group="dataset"
    # ) as run:
    #     run.use_artifact("ohlc_data_prototype_view:latest")

    #     y_train, y_test, X_train, X_test = prepare_data(data, fh=fh)

    #     for split in ["train", "test"]:
    #         split_X = locals()[f"X_{split}"]
    #         split_y = locals()[f"y_{split}"]

    #         split_metadata = {
    #             "timespan": [
    #                 split_X.index.get_level_values(-1).min(),
    #                 split_X.index.get_level_values(-1).max(),
    #             ],
    #             "dataset_size": len(split_X),
    #             "num_areas": len(split_X.index.get_level_values(0).unique()),
    #             "num_consumer_types": len(split_X.index.get_level_values(1).unique()),
    #             "y_features": split_y.columns.tolist(),
    #             "X_features": split_X.columns.tolist(),
    #         }
    #         artifact = wandb.Artifact(
    #             name=f"split_{split}",
    #             type="split",
    #             metadata=split_metadata,
    #         )
    #         run.log_artifact(artifact)

    #     run.finish()

    # return y_train, y_test, X_train, X_test


def prepare_data(
    data: pd.DataFrame, target: str = "close", fh: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Structure the OHLC data for training:
    - Set the index as required by sktime (symbol and timestamp).
    - Prepare exogenous variables (open, high, low, volume).
    - Prepare the time series to be forecasted (target: close).
    - Split the data into train and test sets.
    """

    # Set the index with 'symbol' and 'timestamp' as required by sktime.
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index(["symbol", "timestamp"]).sort_index()

    # Prepare exogenous variables (excluding the target variable, e.g., 'close').
    X = data.drop(columns=[target])

    # Prepare the time series to be forecasted (target is 'close').
    y = data[[target]]

    # Split the data into train and test sets using sktime's temporal_train_test_split.
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=fh)

    return y_train, y_test, X_train, X_test



if __name__ == "__main__":
    fire.Fire(load_dataset_from_feature_store)