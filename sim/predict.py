import os
import sys
curDir = os.path.dirname(__file__)
sys.path.append('{0}/../scripts/'.format(curDir))
sys.path.append('{0}/../models/'.format(curDir))

import pandas as pd
import datetime
import numpy as np
from indicators import Indicators
# from auto_sklearn_model import AutoSklearnModel
from svm_model import SVMModel

START_DATE = '2007-01-01'
END_DATE = datetime.date.today()

def create_classify_labels(adj_close_prices, num_days):
    classified = adj_close_prices.rolling(window=num_days+1).\
                    apply(lambda t: 1 if t[num_days] >= t[0] else -1)[num_days:]
    return pd.DataFrame(classified.values, columns=['Labels'])

def get_indicators(stock):
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    indicators = Indicators(stock, start, end)
    return indicators.calculate_all_indicators()

def buy_or_sell(stock, num_days):
    # Get indicators data and labels
    indicators_df_origin = get_indicators(stock)
    train_Y_df = create_classify_labels(indicators_df_origin['Adj Close Price'], num_days)
    train_Y = np.array(train_Y_df.transpose().values[0])
    train_X_df = indicators_df_origin[:len(train_Y_df)]
    test_X_df = indicators_df_origin[len(train_Y_df):]

    # AutoSklearnModel
    # auto_model = AutoSklearnModel()
    # (model, training_score) = auto_model.get_model(train_X, train_Y)
    # predicted = model.predict(test_X)

    # SVM Classification
    svm_model = SVMModel()
    training_score = svm_model.train(train_X_df, train_Y)
    predicted = svm_model.predict(test_X_df)

    dates = pd.to_datetime(test_X_df.index)

    for i in range(len(predicted)):
        print("    {0}".format(dates[i].strftime('%Y-%m-%d')))
        print("    next {0}-day prediction: {1}".format(num_days, predicted[i]))
        print("")

    print("Training data size was: {0}".format(len(train_X_df)))
    print("Training accuracy was: {0:.4f}".format(training_score))
    print("")

    return predicted[0]

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 predict.py <num_days> <stock_symbol>")
        quit()

    buy_or_sell(sys.argv[2], int(sys.argv[1]))
