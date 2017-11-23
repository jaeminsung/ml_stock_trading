import os
import sys
curDir = os.path.dirname(__file__)
sys.path.append('{0}/../scripts/'.format(curDir))

import pandas as pd
import numpy as np
from indicators import Indicators
from auto_sklearn_model import AutoSklearnModel

# start = pd.to_datetime('2012-01-01')
# end = datetime.date.today()
# ind_obj = Indicators('SPY', start, end)

# print(ind_obj.calculate_all_indicators())
# adj = ind_obj.adj_close_price()
# print(utils.create_classify_labels(adj, 2))

START_DATE = '2012-01-01'
END_DATE = '2017-01-01'
STOCK = 'COF'
NUM_DAYS = 10

def create_classify_labels(adj_close_prices, num_days):
    classified = adj_close_prices.rolling(window=num_days+1).\
                    apply(lambda t: 1 if t[num_days] >= t[0] else -1)[num_days:]
    return pd.DataFrame(classified.values, columns=['Labels'])

def get_indicators(stock):
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    indicators = Indicators(stock, start, end)
    return indicators.calculate_all_indicators()

# Get indicators data and labels
indicators_df_origin = get_indicators(STOCK)
labels_df = create_classify_labels(indicators_df_origin['Adj Close Price'], NUM_DAYS)
indicators_df = indicators_df_origin[:len(labels_df)]
X = np.array(indicators_df)
Y = np.array(labels_df.transpose().values[0])

auto_model = AutoSklearnModel()
(model, score) = auto_model.get_model(X, Y)
print(score)
