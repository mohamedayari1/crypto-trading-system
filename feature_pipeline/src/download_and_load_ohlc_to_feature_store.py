import logging
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
import hopsworks
import fire

from paths import DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dataset_generation')

# List of important cryptocurrencies (can be extended as needed)
IMPORTANT_CRYPTOCURRENCIES = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "LTC-USD"]

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

    # create list of days as strings
    logger.info(f"Generating list of days from {from_day} to {to_day}")
    days = pd.date_range(start=from_day, end=to_day, freq="1D")
    days = [day.strftime("%Y-%m-%d") for day in days]

    # create empty dataframe
    data = pd.DataFrame()

    # create download dir folder if it doesn't exist
    if not (DATA_DIR / 'downloads').exists():
        logger.info('Creating directory for downloads')
        (DATA_DIR / 'downloads').mkdir(parents=True)

    for product_id in product_ids:
        for day in days:
            # download file if it doesn't exist
            file_name = DATA_DIR / 'downloads' / f'{product_id}_{day}.csv'
            if file_name.exists():
                logger.info(f'File {file_name} already exists, skipping download')
                data_one_day = pd.read_csv(file_name)
            else:
                logger.info(f'Downloading data for {product_id} on {day}')
                data_one_day = download_data_for_one_day(product_id, day)
                data_one_day.to_csv(file_name, index=False)
                logger.info(f"Data for {day} saved to {file_name}")

            # combine today's file with the rest of the data
            data = pd.concat([data, data_one_day])

    # save data to disk as CSV
    output_file = DATA_DIR / "ohlc_data_last_7_days.csv"
    data.to_csv(output_file, index=False)
    logger.info(f"All data saved to {output_file}")

    # Load data to feature store
    load_data_to_feature_store(data)
    
    return output_file

def download_data_for_one_day(product_id: str, day: str) -> pd.DataFrame:
    """
    Downloads one day of data and returns pandas DataFrame
    """
    # create start and end date strings
    start = f'{day}T00:00:00'
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    end = f'{end}T00:00:00'

    # call API
    URL = f'https://api.exchange.coinbase.com/products/{product_id}/candles?start={start}&end={end}&granularity=3600'
    logger.info(f"Requesting data from Coinbase API for {product_id} on {day}")
    r = requests.get(URL)
    data = r.json()

    # transform list of lists to pandas dataframe and return
    return pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

def load_data_to_feature_store(data: pd.DataFrame):
    """
    Loads the OHLC data into the Hopsworks Feature Store
    """
    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=FS_API_KEY, project=FS_PROJECT_NAME)
    
    try:
        # Get feature store
        feature_store = project.get_feature_store()
        
        # Add product_id column before preparing data
        data['product_id'] = data.apply(lambda x: x.name[0] if isinstance(x.name, tuple) else 'Unknown', axis=1)
        
        # Prepare data for feature store
        data['timestamp'] = pd.to_datetime(data['time'], unit='s')
        data['symbol'] = data['product_id'].str.split('-').str[0]  # Get the currency symbol (e.g., 'BTC')
        
        # Drop unnecessary columns and reorder
        data = data.drop(columns=['time', 'product_id'])
        data = data[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Define feature group schema
        feature_group = feature_store.get_or_create_feature_group(
            name="ohlc_data",
            version=1,
            description="OHLC data for cryptocurrencies",
            primary_key=["symbol"],  # Remove timestamp from primary key
            online_enabled=True,
            event_time='timestamp',
            statistics_config={
                "enabled": True,
                "correlations": True,
                "histograms": True
            }
        )

        # Load data into feature store
        feature_group.insert(data, write_options={"wait_for_job": True})
        
        logger.info("Data successfully loaded to feature store!")
        
    except Exception as e:
        logger.error(f"Error loading data to feature store: {str(e)}")
        raise

if __name__ == '__main__':
    fire.Fire(download_ohlc_data_from_coinbase)
