
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn import linear_model

import rqdatac
rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))[::-1]


def get_stock_beta(benchmark, date):

    exp_weight = get_exponential_weight(half_life=63, length=252)

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_252_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-252]

    stock_list = rqdatac.all_instruments(type = 'CS', date = latest_trading_date)['order_book_id'].values.tolist()

    stock_daily_return = rqdatac.get_price(stock_list, rqdatac.get_previous_trading_date(latest_trading_date), trading_date_252_before, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据少于66个的股票

    inds = stock_daily_return.isnull().sum()[stock_daily_return.isnull().sum() > (len(stock_daily_return) - 66)].index

    filtered_stock_daily_return = stock_daily_return.drop(inds, axis=1)

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=latest_trading_date, end_date=trading_date_252_before, tenor='3M')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[filtered_stock_daily_return.index]

    stock_excess_return = filtered_stock_daily_return.T.subtract(risk_free_return.iloc[:,0]).T

    market_portfolio_daily_return = rqdatac.get_price(benchmark, rqdatac.get_previous_trading_date(latest_trading_date), trading_date_252_before, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    market_portfolio_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:,0])

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_market_portfolio_variance = weighted_market_portfolio_excess_return.var()

    beta = [weighted_market_portfolio_excess_return.cov(weighted_stock_excess_return[stock]) / weighted_market_portfolio_variance for stock in stock_excess_return.columns]

    stock_beta = pd.Series(beta, index=stock_excess_return.columns)

    return stock_beta


