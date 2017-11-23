import os
import pandas as pd

def write_csv_file(data_df, stock, filename, include_header=False):
    curDir = os.path.dirname(__file__)
    stock_data_dir = 'data/{}'.format(stock)
    filepath = os.path.join(curDir, os.pardir, stock_data_dir)

    if not os.path.exists(filepath):
        os.mkdir(filepath)
    path_to_write = os.path.join(filepath, filename)
    data_df.to_csv(path_to_write, index=False, header=include_header)

def read_stock_data(stock):
    curDir = os.path.dirname(__file__)
    stock_data_dir = 'data/{0}/{0}_data.csv'.format(stock)
    csv_filepath = os.path.join(curDir, os.pardir, stock_data_dir)
    return pd.read_csv(csv_filepath)

def include_n_days_before(data_df, num_days, start_date, end_date):
    # NEED TO HANDLE ERROR IF START_DATE >= END_DATE
    str_start_date = start_date.strftime('%Y-%m-%d')
    str_end_date = end_date.strftime('%Y-%m-%d')
    data_df = data_df[data_df['Date'] <= str_end_date]
    ind = data_df[data_df['Date'] >= str_start_date].index.tolist()[0]
    return data_df[ind-num_days+1:len(data_df)]
