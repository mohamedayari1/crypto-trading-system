import lightgbm as lgb
from sktime.forecasting.compose import make_reduction, ForecastingPipeline
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer
import pandas as pd

# Assuming 'OHLC_data' is your DataFrame containing Open, High, Low, Close prices

def build_model(config: dict):
    """
    Build an Sktime model for forecasting stock Close prices based on OHLC data.

    The model will summarize rolling windows of OHLC data and forecast the Close price.
    """
    
    # Configuration for window summarization
    lag = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__lag",
        list(range(1, 72 + 1)),  # Use past 72 data points (e.g., days or minutes)
    )
    mean = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__mean",
        [[1, 24], [1, 48], [1, 72]],  # Rolling mean over different windows
    )
    std = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__std",
        [[1, 24], [1, 48], [1, 72]],  # Rolling standard deviation over different windows
    )
    n_jobs = config.pop("forecaster_transformers__window_summarizer__n_jobs", 1)
    
    # Use WindowSummarizer to create lag, mean, and std features from the past data
    window_summarizer = WindowSummarizer(
        **{"lag_feature": {"lag": lag, "mean": mean, "std": std}},
        n_jobs=n_jobs,
    )
    
    # Use LGBMRegressor as the forecasting model
    regressor = lgb.LGBMRegressor()
    
    # Create a forecasting pipeline using 'make_reduction' for regression task
    forecaster = make_reduction(
        regressor,
        transformers=[window_summarizer],
        strategy="recursive",
        pooling="global",
        window_length=None,
    )
    
    # Forecasting pipeline with date-time features and window summarizer
    pipe = ForecastingPipeline(
        steps=[
            ("forecaster", forecaster),
            ("daily_season", DateTimeFeatures(
                manual_selection=["day_of_week", "hour_of_day"],
                keep_original_columns=True,
            )),
        ]
    )
    
    pipe = pipe.set_params(**config)

    return pipe



def build_baseline_model(seasonal_periodicity: int):
    """Builds a naive forecaster baseline model using Sktime that predicts the last value given a seasonal periodicity."""

    return NaiveForecaster(sp=seasonal_periodicity)
