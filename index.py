from sim.predict import buy_or_sell

if __name__ == "__main__":
    stock = 'SPY'
    num_days = 10

    print(buy_or_sell(stock, num_days))
