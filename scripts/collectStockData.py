import datetime
import sys
import requests
import pandas as pd
from pandas.tseries.offsets import BDay
import utils

# Constants
API_USERNAME = '' # Provide your own API username and password
API_PASSWORD = ''
INTRINIO_URL = 'https://api.intrinio.com/prices'
COLUMN_NAMES = ['Date', 'Symbol', 'Open', 'Close', 'High', 'Low', 'Volume']

def call_intrinio_api(stock, api_data, page_number, start, end):
    url = '{0}?identifier={1}&start_date={2}&end_date={3}&sort_order=asc&page_number={4}'\
            .format(INTRINIO_URL, stock, start, end, page_number)
    response = requests.get(url, auth=(API_USERNAME, API_PASSWORD), timeout=10)

    if response.status_code == requests.codes.ok:
        response_json = response.json()
        data = response_json['data']
        total_pages = response_json['total_pages']

        for row in data:
            data_feed = [row['date'], stock, row['adj_open'], \
                        row['adj_close'], row['adj_high'], row['adj_low'], \
                        row['adj_volume']]
            feed_df = pd.DataFrame([data_feed], columns=COLUMN_NAMES)
            api_data = api_data.append(feed_df, ignore_index=True)
        return (api_data, total_pages)
    else:
        return (api_data, 0)

def collect_data(stock, start_date=datetime.date(2007, 1, 1), end_date=datetime.date.today()):
    start_date = start_date - BDay(50+1)
    page_num = 1
    api_data = pd.DataFrame(columns=COLUMN_NAMES)
    (api_data, total_pages) = call_intrinio_api(stock, api_data, page_num, start_date, end_date)

    if total_pages > 1:
        while page_num < total_pages:
            page_num += 1
            (api_data, total_pages) = \
                call_intrinio_api(stock, api_data, page_num, start_date, end_date)

    return api_data

def write_csv_data(stock, data_df):
    filename = '{0}_data.csv'.format(stock)
    utils.write_csv_file(data_df, stock, filename, include_header=True)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python collectStockData.py <i>")
        print("i: stock symbol to query market data for")
        quit()

    stock_symbol = sys.argv[1]

    result = collect_data(stock_symbol)
    print(result)
    if len(result) > 0:
        write_csv_data(stock_symbol, result)
