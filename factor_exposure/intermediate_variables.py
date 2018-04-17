import numpy as np
import pandas as pd
import datetime

import rqdatac

rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))



def get_exponential_weight(half_life, length):

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))




def get_stock_daily_excess_return(stock_list, benchmark_id, start_date, end_date):

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    stocks_daily_return = rqdatac.get_price(stock_list, start_date=start_date, end_date=end_date, frequency='1d', fields='ClosingPx', adjust_type='pre').fillna(method='ffill').pct_change()[1:]

    benchmark_daily_return = rqdatac.get_price(benchmark_id, start_date=start_date, end_date=end_date, fields='ClosingPx', adjust_type='pre').fillna(method='ffill').pct_change()[1:]

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=start_date, end_date=end_date, tenor='0S')

    daily_risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 240)) - 1)

    return stocks_daily_return - daily_risk_free_return, benchmark_daily_return - daily_risk_free_return


def get_market_portfolio_daily_excess_return(stock_list, start_date, end_date):

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    stocks_daily_return = rqdatac.get_price(stock_list, start_date=start_date, end_date=end_date, frequency='1d', fields='ClosingPx', adjust_type='pre').fillna(method='ffill').pct_change()[1:]

    benchmark_daily_return = rqdatac.get_price(benchmark_id, start_date=start_date, end_date=end_date, fields='ClosingPx', adjust_type='pre').fillna(method='ffill').pct_change()[1:]


    return stocks_daily_return - daily_risk_free_return, benchmark_daily_return - daily_risk_free_return


def get_daily_risk_free_return(start_date, end_date):

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=start_date, end_date=end_date, tenor='0S')

    daily_risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 240)) - 1)

    return daily_risk_free_return


def benchmark_beta_exposure():
    ### load listed stocks ###

    # We need 21 extra trading days for short-term volalitity calculation.

    complete_path = os.path.join(temp_path, "df_listed_stocks_with_21_extra_trading_days.pkl")

    pkfl = open(complete_path, 'rb')

    listed_stocks = pickle.load(pkfl)

    pkfl.close()

    ### load benchmark daily excess return rate ###

    complete_path = os.path.join(temp_path, "benchmark_daily_excess_return_rate.pkl")

    pkfl = open(complete_path, 'rb')

    benchmark_daily_excess_return_rate = pickle.load(pkfl)

    pkfl.close()

    ### load stock daily excess return rate ###

    complete_path = os.path.join(temp_path, "stock_daily_excess_return_rate.pkl")

    pkfl = open(complete_path, 'rb')

    stock_daily_excess_return_rate = pickle.load(pkfl)

    pkfl.close()

    # Take the transpose of listed_stocks to facilitate the calculation.

    listed_stocks_t = listed_stocks.transpose()

    # remove the "%H-%M-%S" in the time-index

    listed_stocks.index = [datetime.datetime.strptime(i.strftime("%Y-%m-%d"), "%Y-%m-%d").date() for i in
                           listed_stocks.index]

    for date in listed_stocks.index:

        benchmark_beta = []

        # Obtain the order_book_id list of listed stock at current trading day.

        stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()

        # subset stocks' time serie for the past 200 trading days for benchmark beta estimation

        stock_today = stock_daily_excess_return_rate.index.get_loc(str(date))

        stock_today = stock_daily_excess_return_rate.index[stock_today]

        stock_two_hundreds_days_before = stock_daily_excess_return_rate.index.get_loc(str(date)) - 199

        stock_two_hundreds_days_before = stock_daily_excess_return_rate.index[stock_two_hundreds_days_before]

        series_stock_return_rate = stock_daily_excess_return_rate[stock_list].ix[
                                   stock_two_hundreds_days_before: stock_today].copy()

        # subset benchmark' series

        benchmark_today = benchmark_daily_excess_return_rate.index.get_loc(date)

        benchmark_today = benchmark_daily_excess_return_rate.index[benchmark_today]

        benchmark_two_hundreds_days_before = benchmark_daily_excess_return_rate.index.get_loc(
            date) - 199  ## Using the past 200 trading days to estimate benchmark beta.

        benchmark_two_hundreds_days_before = benchmark_daily_excess_return_rate.index[
            benchmark_two_hundreds_days_before]

        series_benchmark_return_rate = benchmark_daily_excess_return_rate.ix[
                                       benchmark_two_hundreds_days_before: benchmark_today].copy()

        for stock in series_stock_return_rate.columns:

            # skip the calcualtion if the stock has more than 68 NANs, i.e., at least 132 trading days (6 months) data available.

            if series_stock_return_rate[stock].isnull().sum() > 68:

                benchmark_beta.append(np.nan)

            else:

                # merge them into a dataframe so that we can call covariance calculation in Pandas, which is NaN-friendly.

                df_excess_return_rate = pd.DataFrame(
                    [series_stock_return_rate[stock].values, series_benchmark_return_rate.values],
                    index=['stock', 'benchmark'])

                df_excess_return_rate = pd.DataFrame.transpose(df_excess_return_rate)

                # Note that in NumPy, the variance/coriance are calculated in a biased way, i.e., with degree of freedom = 0.

                benchmark_beta.append(df_excess_return_rate.cov().iloc[0, 1] / series_benchmark_return_rate.var())

        if date == listed_stocks.index[0]:

            # Create the dataframe for benchmark beta.

            df_benchmark_beta = pd.DataFrame(benchmark_beta, index=series_stock_return_rate.columns, columns=[date])

        else:

            benchmark_beta = pd.DataFrame(benchmark_beta, index=series_stock_return_rate.columns, columns=[date])

            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.

            df_benchmark_beta = pd.concat([df_benchmark_beta, benchmark_beta], axis=1, join='outer')

    df_benchmark_beta = df_benchmark_beta.transpose()

    return df_benchmark_beta


### function for winsorization and standardization of atomic descriptor ###

def winsorization_and_standardization(df_atomic_descriptor):
    #### Standardization ###

    std_df_atomic_descriptor = (df_atomic_descriptor - df_atomic_descriptor.mean()) / df_atomic_descriptor.std()

    #### Winsorization ###

    sd_atomic_descriptor = std_df_atomic_descriptor.std()

    mean_atomic_descriptor = std_df_atomic_descriptor.mean()

    upper_limit = mean_atomic_descriptor + 3 * sd_atomic_descriptor

    lower_limit = mean_atomic_descriptor - 3 * sd_atomic_descriptor

    # Replace the outleirs

    std_df_atomic_descriptor[
        (std_df_atomic_descriptor > upper_limit) & (std_df_atomic_descriptor != np.nan)] = upper_limit

    std_df_atomic_descriptor[
        (std_df_atomic_descriptor < lower_limit) & (std_df_atomic_descriptor != np.nan)] = lower_limit

    return std_df_atomic_descriptor


