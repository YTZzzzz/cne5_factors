import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn import linear_model
from operators import individual_factor_imputation
from intermediate_variables import *


import rqdatac
rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))


def get_stock_beta(stock_list, stock_excess_return, benchmark, latest_trading_date, market_cap_on_current_day):

    trading_date_253_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-253]

    exp_weight = get_exponential_weight(half_life = 63, length = 252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_253_before, end_date=latest_trading_date, tenor='3M')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[stock_excess_return.index]

    market_portfolio_daily_return = rqdatac.get_price(benchmark, trading_date_253_before,latest_trading_date,frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    market_portfolio_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:,0])

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_market_portfolio_variance = weighted_market_portfolio_excess_return.var()

    beta = [weighted_market_portfolio_excess_return.cov(weighted_stock_excess_return[stock])/weighted_market_portfolio_variance for stock in stock_excess_return.columns]

    stock_beta = pd.Series(beta, index = stock_excess_return.columns)

    # 用回归方法处理 beta 的缺失值

    imputed_stock_beta = individual_factor_imputation(stock_list, stock_beta, market_cap_on_current_day, latest_trading_date.strftime('%Y-%m-%d'))

    return imputed_stock_beta

