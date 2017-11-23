import os
import sys
curDir = os.path.dirname(__file__)
sys.path.append('{0}/../scripts/'.format(curDir))
sys.path.append('{0}/../models/'.format(curDir))

import datetime
import pandas as pd
import numpy as np
from indicators import Indicators
# from auto_sklearn_model import AutoSklearnModel
from svm_model import SVMModel
import sklearn.metrics

START_DATE = '2007-01-01'
SPLIT_DATE = '2016-01-01'
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

def calculate_return(decision, holdingPos, curPrice, buyPrice):
    return_profit = 0.0
    changedPos = holdingPos

    if decision == -1 and holdingPos:
        return_profit = curPrice - buyPrice
        print("    sell at {0}; made {1} per share".format(curPrice, return_profit))
        changedPos = not holdingPos
    elif decision == 1 and not holdingPos:
        print("    buy at {0}".format(curPrice))
        buyPrice = curPrice
        changedPos = not holdingPos
    else:
        print("    do nothing")

    return (changedPos, buyPrice, return_profit)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 simulate.py <num_days> <stock_symbol>")
        quit()

    NUM_DAYS = int(sys.argv[1])
    STOCK = sys.argv[2]

    # Get indicators data and labels
    indicators_df_origin = get_indicators(STOCK)
    labels_df = create_classify_labels(indicators_df_origin['Adj Close Price'], NUM_DAYS)
    indicators_df = indicators_df_origin[:len(labels_df)]

    split_date = pd.to_datetime(SPLIT_DATE)

    train_X_df = indicators_df[pd.to_datetime(indicators_df.index) < split_date]
    test_X_df = indicators_df[pd.to_datetime(indicators_df.index) >= split_date]

    labels = np.array(labels_df.transpose().values[0])
    train_Y = labels[0:len(train_X_df)]
    test_Y = labels[len(train_X_df):]

    # AutoSklearnModel
    # auto_model = AutoSklearnModel()
    # (model, training_score) = auto_model.get_model(train_X, train_Y)
    # predicted = model.predict(test_X)

    # SVM Classification
    svm_model = SVMModel()
    training_score = svm_model.train(train_X_df, train_Y)
    predicted = svm_model.predict(test_X_df)

    dates = pd.to_datetime(test_X_df.index)
    close_prices = test_X_df['Adj Close Price']

    n_bins = []
    for n in range(NUM_DAYS):
        n_bins.append({'holding': False, 'num_shares': 1.0, 'buy_price': 0.0})
    total_profit = 0.0

    print("Start Simulation")
    print("")

    for i in range(len(predicted)):
        bin_number = i % NUM_DAYS
        holding = n_bins[bin_number]['holding']
        # num_shares = n_bins[bin_number]['num_shares']
        buy_price = n_bins[bin_number]['buy_price']
        print("    {0}".format(dates[i].strftime('%Y-%m-%d')))
        print("    next {0}-day prediction: {1}".format(NUM_DAYS, predicted[i]))
        holding, buy_price, profit = \
            calculate_return(predicted[i], holding, close_prices[i], buy_price)
        total_profit += profit
        n_bins[bin_number]['holding'] = holding
        n_bins[bin_number]['buy_price'] = buy_price
        print("")

    print("End Simulation")
    print("")

    first_day = dates[0].strftime('%Y-%m-%d')
    first_day_price = close_prices[0]

    last_day = dates[-1].strftime('%Y-%m-%d')
    last_day_price = close_prices[-1]


    print("From {0} to {1}, {2} stock price changed from ${3:.2f} to ${4:.2f}, ${5:.2f} per share ({6:.2f}%)"\
    .format(first_day, last_day, STOCK, first_day_price, last_day_price, \
    last_day_price-first_day_price, ((last_day_price-first_day_price)/first_day_price)*100))

    print("Over the same period, total profit made using the ML model was ${0:.2f} per share ({1:.2f}%)"\
    .format(total_profit, (total_profit/first_day_price)*100))

    score = sklearn.metrics.accuracy_score(test_Y, predicted)
    print("Prediction accuracy of the ML model was: {0:.4f}".format(score))
    print("Training accuracy was: {0:.4f}".format(training_score))
    print("")
