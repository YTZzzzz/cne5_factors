
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn import linear_model
from intermediate_variables import *

import rqdatac
rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))


def get_stock_beta(stock_list,benchmark, date):

    exp_weight = get_exponential_weight(half_life=63, length=252)

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_253_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-253]

    stock_daily_return = rqdatac.get_price(stock_list, trading_date_253_before,latest_trading_date,frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据少于66个的股票

    inds = stock_daily_return.isnull().sum()[stock_daily_return.isnull().sum() > (len(stock_daily_return) - 66)].index

    filtered_stock_daily_return = stock_daily_return.drop(inds, axis=1)

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_253_before, end_date=latest_trading_date, tenor='3M')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[filtered_stock_daily_return.index]

    stock_excess_return = filtered_stock_daily_return.T.subtract(risk_free_return.iloc[:,0]).T

    market_portfolio_daily_return = rqdatac.get_price(benchmark, trading_date_253_before,latest_trading_date,frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    market_portfolio_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:,0])

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_market_portfolio_variance = weighted_market_portfolio_excess_return.var()

    beta = [weighted_market_portfolio_excess_return.cov(weighted_stock_excess_return[stock]) / weighted_market_portfolio_variance for stock in stock_excess_return.columns]

    stock_beta = pd.Series(beta, index=stock_excess_return.columns)

    return stock_beta


'''
# test beta

# 沪深300作为投资组合 中证500作为benchmark

# 投资组合beta:
date = '2017-03-03'

latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

trading_date_253_before = \
rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-253]

stock_list = rqdatac.all_instruments(type='CS', date=latest_trading_date)['order_book_id'].values.tolist()

portfolio_daily_return = rqdatac.get_price('000300.XSHG', trading_date_253_before,latest_trading_date,frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_253_before, end_date=latest_trading_date,
                                                      tenor='3M')
risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[portfolio_daily_return.index]

market_portfolio_daily_return = rqdatac.get_price('000905.XSHG', trading_date_253_before,latest_trading_date,frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

market_portfolio_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:, 0])

portfolio_excess_return = portfolio_daily_return.subtract(risk_free_return.iloc[:, 0])

exp_weight = get_exponential_weight(half_life=63, length=252)

weighted_portfolio_excess_return = portfolio_excess_return.T.multiply(exp_weight).T

weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

weighted_market_portfolio_variance = weighted_market_portfolio_excess_return.var()

portfolio_beta = weighted_market_portfolio_excess_return.cov(weighted_portfolio_excess_return) / weighted_market_portfolio_variance

benchmark = '000905.XSHG'

weight = rqdatac.index_weights('000300.XSHG',date)

stock_beta = get_stock_beta(stock_list,benchmark,date)

test_beta = (stock_beta[weight.index] * weight).sum()
'''
