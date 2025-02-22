import hopsworks
import pandas as pd
from great_expectations.core import ExpectationSuite
from hsfs.feature_group import FeatureGroup

def to_feature_store(
    data: pd.DataFrame,
    validation_expectation_suite: ExpectationSuite,
    feature_group_version: int,
) -> FeatureGroup:
    """
    This function takes in OHLC data as a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a feature store.
    """

    # Connect to Hopsworks Feature Store
    project = hopsworks.login(
        api_key_value="Pg6hKTom6MjSnbNF.VBmTgApGW1pB8bCTK0MPGoixrydLJ9V92nrOjC4wAsfGnnu8ZOC7fFR3iUJL2qQQ",
        project="crypto_trading_system"
    )
    feature_store = project.get_feature_store()

    # Create feature group for OHLC data
    ohlc_feature_group = feature_store.get_or_create_feature_group(
        name="ohlc_data",
        version=feature_group_version,
        description="OHLC data for cryptocurrency trading",
        primary_key=["timestamp", "symbol"],  # Example of primary key columns
        event_time="timestamp",  # Column representing the event time (timestamp)
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    # Upload OHLC data to feature store
    ohlc_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions
    feature_descriptions = [
        {
            "name": "timestamp",
            "description": """
                        The timestamp represents the time the OHLC data was recorded in UTC.
                        """,
            "validation_rules": "Always full timestamp format (YYYY-MM-DD HH:MM:SS)",
        },
        {
            "name": "symbol",
            "description": """
                        The symbol for the cryptocurrency, e.g., 'BTC', 'ETH', etc.
                        """,
            "validation_rules": "String representing the cryptocurrency symbol.",
        },
        {
            "name": "open",
            "description": """
                        The opening price for the given time interval.
                        """,
            "validation_rules": ">=0 (float)",
        },
        {
            "name": "high",
            "description": """
                        The highest price for the given time interval.
                        """,
            "validation_rules": ">=0 (float)",
        },
        {
            "name": "low",
            "description": """
                        The lowest price for the given time interval.
                        """,
            "validation_rules": ">=0 (float)",
        },
        {
            "name": "close",
            "description": """
                        The closing price for the given time interval.
                        """,
            "validation_rules": ">=0 (float)",
        },
        {
            "name": "volume",
            "description": """
                        The trading volume for the given time interval.
                        """,
            "validation_rules": ">=0 (float)",
        },
    ]

    for description in feature_descriptions:
        ohlc_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics configuration and compute statistics
    ohlc_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    ohlc_feature_group.update_statistics_config()
    ohlc_feature_group.compute_statistics()

    return ohlc_feature_group
