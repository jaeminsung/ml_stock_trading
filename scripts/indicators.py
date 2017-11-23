import numpy as np
import pandas as pd
import utils

class Indicators:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.data = utils.read_stock_data(stock)

    def calculate_all_indicators(self):
        indicators = [
            self.adj_close_price(),
            self.bollinger_bands(),
            self.cci(4),
            self.cci(12),
            self.cci(20),
            self.ema(2),
            self.ema(6),
            self.ema(10),
            self.ema(12),
            self.macd(),
            self.mfi(14),
            self.mfi(16),
            self.mfi(18),
            self.obv(),
            self.px_volume(),
            self.rsi(6),
            self.rsi(12),
            self.sma(3),
            self.sma(10),
            self.trix(),
            self.volatility(2),
            self.volatility(4),
            self.volatility(6),
            self.volatility(8),
            self.volatility(10),
            self.volatility(12),
            self.volatility(14),
            self.volatility(16),
            self.volatility(18),
            self.volatility(20),
            self.willr()
            ]
        dates = utils.include_n_days_before(self.data, 1, self.start_date, self.end_date)['Date']
        df = pd.concat(indicators, axis=1)
        return df.set_index(dates)

    def adj_close_price(self):
        df = utils.include_n_days_before(self.data, 1, self.start_date, self.end_date)
        return pd.DataFrame(df['Close'].values, columns=['Adj Close Price'])

    def bollinger_bands(self):
        window_len = 20

        def Bollinger_Bands(stock_price, window_size, num_of_std):
            rolling_mean = stock_price['Close'].rolling(window=window_size).mean()[window_size-1:]
            rolling_std = stock_price['Close'].rolling(window=window_size).std()[window_size-1:]
            upper_band = np.add(rolling_mean, rolling_std * num_of_std)
            lower_band = np.subtract(rolling_mean, rolling_std * num_of_std)
            return rolling_mean, upper_band, lower_band

        prices = utils.include_n_days_before(self.data, window_len, self.start_date, self.end_date)
        middle, upper, lower = Bollinger_Bands(prices, window_len, 2)
        result_df = pd.DataFrame({'BB_Middle': middle.values, \
                                  'BB_Upper': upper.values, 'BB_Lower': lower.values})
        return result_df

    def cci(self, num_days):
        df = utils.include_n_days_before(self.data, num_days, self.start_date, self.end_date)
        df_after_start_date = df[num_days-1:]

        def calculate_tp(t):
            return(t['High']+t['Low']+t['Close'])/3

        tp_df = df_after_start_date.apply(calculate_tp, 1)

        # calculate TpAvg(t) where TpAvg(t,n)=Avg(Tp(t)) over [t, t-1, . . . , t-n+1];
        tp_avg_df = df.apply(calculate_tp, 1)
        tp_avg_df = tp_avg_df.rolling(window=num_days).mean()
        tp_avg_df = tp_avg_df[(num_days-1):]

        # calculate MD(t) where MD(t)=Avg(Abs(Tp(t)-TpAvg(t,n)));
        md = np.mean(np.absolute(np.subtract(tp_df, tp_avg_df)))

        # calculate CCI(t) where CCI(t) = Tp(t)-TpAvg(t,n)/(0.15*MD(t));
        cci = np.subtract(tp_df, tp_avg_df)/(0.15*md)

        return pd.DataFrame(cci.values, columns=['CCI_{0}'.format(num_days)])

    def ema(self, num_days):
        df = utils.include_n_days_before(self.data, num_days, self.start_date, self.end_date)
        ema = df['Close'].ewm(span=num_days).mean()
        ema = ema[num_days-1:]
        return pd.DataFrame(ema.values, columns=['EMA_{0}'.format(num_days)])

    def macd(self):
        n_slow = 26
        n_fast = 12
        n_signal = 9

        df = utils.include_n_days_before(self.data, 1, self.start_date, self.end_date)

        # Calculate MACD
        ema_slow = df['Close'].ewm(span=n_slow, min_periods=1).mean()
        ema_fast = df['Close'].ewm(span=n_fast, min_periods=1).mean()
        macd = np.subtract(ema_fast, ema_slow)

        # Calculate MACD signal
        macd_signal = macd.ewm(span=n_signal, min_periods=1).mean()

        # Calculate MACD histogram
        macd_hist = np.subtract(macd, macd_signal)

        result_df = pd.DataFrame({'MACD': macd.values, \
                                  'MACD_Sig': macd_signal.values, \
                                  'MACD_Hist': macd_hist.values})
        return result_df

    def mfi(self, num_days):
        df = utils.include_n_days_before(self.data, num_days, self.start_date, self.end_date)

        def Money_Flow_Index(window_df, tp_df, mf_df):
            pos_mf = 0.0
            neg_mf = 0.0
            for i in range(len(window_df)):
                tp = tp_df.iloc[i].item()
                mf = mf_df.iloc[i].item()
                if i == 0:
                    pos_mf += mf
                else:
                    tp_before = tp_df.iloc[i-1].item()
                    if tp > tp_before:
                        pos_mf += mf
                    elif tp < tp_before:
                        neg_mf += mf
            mfi = (pos_mf / (pos_mf + neg_mf)) * 100
            return mfi

        tp_df = (df['High']+df['Low']+df['Close'])/3
        mf_df = tp_df * df['Volume']
        col_name = 'MFI_{0}'.format(num_days)
        mfi_df = pd.DataFrame(columns=[col_name])
        for i in range(len(df)-num_days+1):
            temp_df = df.iloc[i:i+num_days, :]
            temp_tp_df = tp_df.iloc[i:i+num_days]
            temp_mf_df = mf_df.iloc[i:i+num_days]
            mfi = Money_Flow_Index(temp_df, temp_tp_df, temp_mf_df)
            mfi_df = mfi_df.append(pd.DataFrame([mfi], columns=[col_name]), ignore_index=True)

        return mfi_df

    def momentum(self, num_days):
        df = utils.include_n_days_before(self.data, num_days+1, self.start_date, self.end_date)
        momentum = df['Close'].rolling(window=num_days+1)\
                    .apply(lambda t: t[num_days]-t[0])
        momentum = momentum[num_days:]
        return pd.DataFrame(momentum.values, columns=['MOM_{0}'.format(num_days)])

    def obv(self):
        df = utils.include_n_days_before(self.data, 1, self.start_date, self.end_date)
        obv_df = pd.DataFrame([0.0], columns=['OBV'])
        obv = 0.0
        for i in range(len(df)-1):
            row_i = df.iloc[i]
            row_i_1 = df.iloc[i+1]
            volume = 0.0
            if row_i_1['Close'] > row_i['Close']:
                volume = row_i_1['Volume']
            elif row_i_1['Close'] < row_i['Close']:
                volume = row_i_1['Volume'] * -1
            obv += volume
            obv_df = obv_df.append(pd.DataFrame([obv], columns=['OBV']), ignore_index=True)
        return obv_df

    def px_volume(self):
        df = utils.include_n_days_before(self.data, 1, self.start_date, self.end_date)
        df = df['Volume']
        return pd.DataFrame(df.values, columns=['PX Volume'])

    def rsi(self, num_days):
        df = utils.include_n_days_before(self.data, num_days+1, self.start_date, self.end_date)
        diff_df = df['Close'].diff()
        diff_df = diff_df[1:]
        avg_up = diff_df.where(lambda x: x > 0, other=0.0)\
                .rolling(window=num_days, min_periods=num_days).mean()
        avg_down = diff_df.where(lambda x: x < 0, other=0.0).abs()\
                .rolling(window=num_days, min_periods=num_days).mean()
        rsi = (avg_up / (avg_up + avg_down)) * 100
        rsi = rsi[num_days-1:]
        return pd.DataFrame(rsi.values, columns=['RSI_{0}'.format(num_days)])

    def sma(self, num_days):
        df = utils.include_n_days_before(self.data, num_days, self.start_date, self.end_date)
        sma = df['Close'].rolling(window=num_days).mean()
        sma = sma[num_days-1:]
        return pd.DataFrame(sma.values, columns=['SMA_{0}'.format(num_days)])

    def trix(self):
        current_df = utils.include_n_days_before(self.data, 1, self.start_date, self.end_date)
        one_day_before_df = utils.include_n_days_before(self.data, 2, self.start_date, self.end_date)

        def calculate_triple_ema(df):
            i = 0
            while i < 3:
                df = df.ewm(span=12, min_periods=1).mean()
                i += 1
            return df

        # TRIX(t) = TR(t)/TR(t-1) where TR(t)=EMA(EMA(EMA(Price(t)))) over n days period
        tr_t = calculate_triple_ema(current_df['Close'])
        tr_t_1 = calculate_triple_ema(one_day_before_df[0:-1]['Close'])
        trix = np.divide(tr_t, tr_t_1)
        return pd.DataFrame(trix.values, columns=['TRIX'])

    def volatility(self, num_days):
        df = utils.include_n_days_before(self.data, num_days, self.start_date, self.end_date)
        vol = df['Close'].rolling(window=num_days).std()
        vol = vol[num_days-1:]
        return pd.DataFrame(vol.values, columns=['Volatility_{0}'.format(num_days)])

    def willr(self):
        df = utils.include_n_days_before(self.data, 1, self.start_date, self.end_date)
        highest_minus_closed = np.subtract(df['High'], df['Close'])
        highest_minus_lowest = np.subtract(df['High'], df['Low'])
        will_r = np.divide(highest_minus_closed, highest_minus_lowest) * 100
        return pd.DataFrame(will_r.values, columns=['WILLR'])
