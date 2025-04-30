from typing import Tuple, Optional
import hopsworks
import pandas as pd
import numpy as np
import logging

# Constants matching training pipeline
FS_API_KEY = "Pg6hKTom6MjSnbNF.VBmTgApGW1pB8bCTK0MPGoixrydLJ9V92nrOjC4wAsfGnnu8ZOC7fFR3iUJL2qQQ"
FS_PROJECT_NAME = "crypto_trading_system"

logger = logging.getLogger(__name__)

def load_prediction_data(
    feature_view_version: int = 1,
    start_datetime: Optional[pd.Timestamp] = None,
    end_datetime: Optional[pd.Timestamp] = None,
    crypto_symbol: Optional[str] = None,
    target: str = "close",
    fh: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from feature store for making predictions.
    
    Args:
        feature_view_version (int): Version of the feature view to use
        start_datetime (pd.Timestamp, optional): Start time for fetching data
        end_datetime (pd.Timestamp, optional): End time for fetching data
        crypto_symbol (str, optional): Specific cryptocurrency to fetch (if None, fetch all)
        target (str): Target variable to predict (default: "close")
        fh (int): Forecast horizon, used for filtering appropriate amount of data
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X (features) and y (target) formatted for prediction
    """
    # Login to Hopsworks
    project = hopsworks.login(
        api_key_value=FS_API_KEY, project=FS_PROJECT_NAME
    )
    fs = project.get_feature_store()
    
    # Get feature view
    feature_view = fs.get_feature_view(
        name="ohlc_data_prototype_view", version=feature_view_version
    )
    
    # Set default time range if not provided
    if end_datetime is None:
        end_datetime = pd.Timestamp.now()
    if start_datetime is None:
        # Get enough historical data for the model
        start_datetime = end_datetime - pd.Timedelta(hours=fh*3)  # 3x horizon as buffer
    
    # Fetch data with filters if symbol is specified
    if crypto_symbol:
        # Use get_batch_data with filtering
        data = feature_view.get_batch_data(
            start_time=start_datetime,
            end_time=end_datetime,
            filters={"symbol": crypto_symbol}
        )
    else:
        # Fetch all cryptocurrency data
        data = feature_view.get_batch_data(
            start_time=start_datetime,
            end_time=end_datetime
        )

    # Format exactly like in training pipeline
    # Set the index with 'symbol' and 'timestamp'
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index(["symbol", "timestamp"]).sort_index()

    # Prepare exogenous variables and target
    X = data.drop(columns=[target])
    y = data[[target]]

    # Ensure the indices have a frequency - same as in train_baseline_model.py
    X = X.copy()
    y = y.copy()
    
    # Create temporary series to apply asfreq
    temp_series = pd.Series(range(len(X.index.levels[1])), index=X.index.levels[1])
    temp_series = temp_series.asfreq('H')
    
    # Set the levels back
    X.index = X.index.set_levels(temp_series.index, level='timestamp')
    y.index = y.index.set_levels(temp_series.index, level='timestamp')

    logger.info(f"Loaded prediction data with shape X: {X.shape}, y: {y.shape}")
    logger.info(f"Time range: {X.index.get_level_values('timestamp').min()} to {X.index.get_level_values('timestamp').max()}")
    
    return X, y

def predict_with_model(
    model,
    X: pd.DataFrame,
    fh: int = 24
) -> pd.DataFrame:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained forecasting model
        X (pd.DataFrame): Exogenous variables formatted for prediction
        fh (int): Forecast horizon
        
    Returns:
        pd.DataFrame: Model predictions
    """
    # Format forecast horizon as expected by the model
    fh_indices = np.arange(fh) + 1
    
    # Make predictions
    predictions = model.predict(X=X, fh=fh_indices)
    
    return predictions