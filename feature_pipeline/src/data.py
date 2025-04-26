
# import logging
# from typing import Optional
# from pathlib import Path
# from datetime import datetime, timedelta

# import pandas as pd
# import requests
# import fire

# from paths import DATA_DIR

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger('dataset_generation')

# # List of important cryptocurrencies (can be extended as needed)
# IMPORTANT_CRYPTOCURRENCIES = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "LTC-USD"]

# def download_ohlc_data_from_coinbase(
#     product_ids: Optional[list] = IMPORTANT_CRYPTOCURRENCIES,
#     from_day: Optional[str] = None,
#     to_day: Optional[str] = None,
# ) -> Path:
#     """
#     Downloads historical candles from Coinbase API for multiple cryptocurrencies and saves data to disk as CSV.
#     """
#     # If no specific date is provided, use the last 7 days
#     if from_day is None:
#         from_day = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
#     if to_day is None:
#         to_day = datetime.now().strftime("%Y-%m-%d")

#     # create list of days as strings
#     logger.info(f"Generating list of days from {from_day} to {to_day}")
#     days = pd.date_range(start=from_day, end=to_day, freq="1D")
#     days = [day.strftime("%Y-%m-%d") for day in days]

#     # create empty dataframe
#     data = pd.DataFrame()

#     # create download dir folder if it doesn't exist
#     if not (DATA_DIR / 'downloads').exists():
#         logger.info('Creating directory for downloads')
#         (DATA_DIR / 'downloads').mkdir(parents=True)

#     for product_id in product_ids:
#         for day in days:
#             # download file if it doesn't exist
#             file_name = DATA_DIR / 'downloads' / f'{product_id}_{day}.csv'
#             if file_name.exists():
#                 logger.info(f'File {file_name} already exists, skipping download')
#                 data_one_day = pd.read_csv(file_name)
#             else:
#                 logger.info(f'Downloading data for {product_id} on {day}')
#                 data_one_day = download_data_for_one_day(product_id, day)
#                 data_one_day.to_csv(file_name, index=False)
#                 logger.info(f"Data for {day} saved to {file_name}")

#             # combine today's file with the rest of the data
#             data = pd.concat([data, data_one_day])

#     # save data to disk as CSV
#     output_file = DATA_DIR / "ohlc_data_last_7_days.csv"
#     data.to_csv(output_file, index=False)
#     logger.info(f"All data saved to {output_file}")

#     return output_file

# def download_data_for_one_day(product_id: str, day: str) -> pd.DataFrame:
#     """
#     Downloads one day of data and returns pandas DataFrame
#     """
#     # create start and end date strings
#     start = f'{day}T00:00:00'
#     end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
#     end = f'{end}T00:00:00'

#     # call API
#     URL = f'https://api.exchange.coinbase.com/products/{product_id}/candles?start={start}&end={end}&granularity=3600'
#     logger.info(f"Requesting data from Coinbase API for {product_id} on {day}")
#     r = requests.get(URL)
#     data = r.json()

#     # transform list of lists to pandas dataframe and return
#     return pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

# if __name__ == '__main__':
#     fire.Fire(download_ohlc_data_from_coinbase)
