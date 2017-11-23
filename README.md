# ml-stock-project
A Python project for predicting stock market performance using machine learning

Examine `buy_or_sell` function used in `index.py`. This function requires 1) a stock symobl and 2) a number of days, to make a prediction on whether the price of a stock will be higher or lower in the given number of days (with ~65% prediction accuracy).

## Getting Started
### Build your Anaconda environment:
This project environment is built using Anaconda. 

`conda env create environment.yml`

### Collect data for a particular stock symbol:
This project uses [Intrinio API](https://intrinio.com/data/company-financials) to collect daily market data, such as closing price, opening price, volume, and etc. You can sign up for a free account that comes with API request limits. After signing up for an account, provide your username and password in `script/collectStockData.py`.

`python3 scripts/collectStockData.py <stock_symobl>`

### Predict price movement 
Edit `stock` and `num_days` in `index.py` to define which symobl you are predicting its price movement in N number of days for.

`python3 index.py`
