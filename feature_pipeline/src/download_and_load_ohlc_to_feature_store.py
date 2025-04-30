import logging
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
import hopsworks
import fire
from paths import DATA_DIR


from datetime import datetime, timedelta

# Hardcoded dates
end_datetime = datetime(2025, 4, 26)  # April 26, 2025
start_datetime = end_datetime - timedelta(days=7)  # Seven days ago, April 19, 2025

# Specify the datetime format
datetime_format = "%Y-%m-%dT%H:%M"

# Convert start_datetime and end_datetime to the specified format
start_time_str = start_datetime.strftime(datetime_format)
end_time_str = end_datetime.strftime(datetime_format)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dataset_generation')

# List of important cryptocurrencies
IMPORTANT_CRYPTOCURRENCIES = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "LTC-USD"]

# Feature Store settings
FS_API_KEY = "Pg6hKTom6MjSnbNF.VBmTgApGW1pB8bCTK0MPGoixrydLJ9V92nrOjC4wAsfGnnu8ZOC7fFR3iUJL2qQQ"
FS_PROJECT_NAME = "crypto_trading_system"

def download_ohlc_data_from_coinbase(
    product_ids: Optional[list] = IMPORTANT_CRYPTOCURRENCIES,
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
) -> Path:
    """
    Downloads historical candles from Coinbase API for multiple cryptocurrencies and saves data to disk as CSV.
    """
    # If no specific date is provided, use the last 7 days
    if from_day is None:
        from_day = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if to_day is None:
        to_day = datetime.now().strftime("%Y-%m-%d")

    # Create list of days as strings
    logger.info(f"Generating list of days from {from_day} to {to_day}")
    days = pd.date_range(start=from_day, end=to_day, freq="1D").strftime("%Y-%m-%d")

    # Create empty dataframe to hold all data
    data = pd.DataFrame()

    # Create download dir folder if it doesn't exist
    download_dir = DATA_DIR / 'downloads'
    download_dir.mkdir(parents=True, exist_ok=True)

    for product_id in product_ids:
        for day in days:
            file_name = download_dir / f'{product_id}_{day}.csv'
            if not file_name.exists():
                logger.info(f"Downloading data for {product_id} on {day}")
                day_data = download_data_for_one_day(product_id, day)
                day_data.to_csv(file_name, index=False)
                logger.info(f"Data for {day} saved to {file_name}")
            else:
                logger.info(f'File {file_name} already exists, skipping download')
                day_data = pd.read_csv(file_name)

            data = pd.concat([data, day_data])

    # Save data to CSV
    output_file = DATA_DIR / "ohlc_data_last_7_days.csv"
    data.to_csv(output_file, index=False)
    logger.info(f"All data saved to {output_file}")

    # Load data to feature store
    load_data_to_feature_store(data)

    return output_file







def download_data_for_one_day(product_id: str, day: str) -> pd.DataFrame:
    """
    Downloads one day of OHLC data from Coinbase and returns a pandas DataFrame.
    """
    start = f'{day}T00:00:00'
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d") + 'T00:00:00'
    url = f'https://api.exchange.coinbase.com/products/{product_id}/candles?start={start}&end={end}&granularity=3600'
    
    logger.info(f"Requesting data from Coinbase API for {product_id} on {day}")
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['symbol'] = product_id  # Add the symbol column
    
    return df




def load_data_to_feature_store(data: pd.DataFrame) -> None:
    """Loads OHLC data into the Hopsworks Feature Store and creates a Feature View."""
    project = hopsworks.login(api_key_value=FS_API_KEY, project=FS_PROJECT_NAME)
    fs = project.get_feature_store()

    try:
        # Prepare data
        data['timestamp'] = pd.to_datetime(data['time'], unit='s')
        data = data[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]  # Include 'symbol'

        # Insert data into the feature store
        feature_group = fs.get_feature_group("ohlc_data")
        feature_group.insert(data, write_options={"wait_for_job": True})

        logger.info("Data successfully loaded to feature store!")



        feature_view = None
        
        # Create Feature View if not exists
        try:
            feature_view = fs.get_feature_view(name="ohlc_data_prototype_view", version=1)
            logger.info("Feature view already exists.")
        except:
            query = feature_group.select_all()
            feature_view = fs.create_feature_view(
                name="ohlc_data_prototype_view",
                description="OHLC data for cryptocurrencies",
                query=query,
                labels=[]
            )

        # Create training dataset.
        logger.info(
            f"Creating training dataset between {start_datetime} and {end_datetime}."
        )
        
        feature_view.create_training_data(
            description="OHLC training dataset",
            data_format="csv",
            # start_time=start_datetime,
            # end_time=end_datetime,
            write_options={"wait_for_job": True},
            coalesce=False,
        )

        # Save metadata.
        metadata = {
            "feature_view_version": feature_view.version,
            "training_dataset_version": 1,
        }
            
            
            
            
    except Exception as e:
        logger.error(f"Error loading data to feature store: {str(e)}")


if __name__ == "__main__":
    fire.Fire(download_ohlc_data_from_coinbase)
