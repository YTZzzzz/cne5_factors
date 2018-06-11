import sys

sys.path.append("/Users/jjj728/git/cne5_factors/factor_exposure/")

from intermediate_variables import *
from operators import *

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn import linear_model

import rqdatac
rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))
#rqdatac.init('ricequant', '8ricequant8',('q-tools.ricequant.com', 16010))



date = '2018-02-06'

year = pd.Series([48, 36, 24, 12, 0])


def get_style_factors(date):

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_252_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-252]

    stock_list = rqdatac.all_instruments(type = 'CS', date = latest_trading_date)['order_book_id'].values.tolist()

    ### 获取因子计算共用的行情数据和财务数据

    recent_report_type, annual_report_type, market_cap_on_current_day,\
    stock_excess_return, market_portfolio_excess_return, recent_five_annual_shares,\
    last_reported_non_current_liabilities, last_reported_preferred_stock = get_financial_and_market_data(stock_list, latest_trading_date, trading_date_252_before)


    ### 细分因子计算

    size_exposure = get_size(market_cap_on_current_day)

    non_linear_size_exposure = get_non_linear_size(size_exposure, market_cap_on_current_day)

    market_portfolio_beta, market_portfolio_beta_exposure = get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day)

    residual_volatility_exposure = get_residual_volatility(stock_list, latest_trading_date, stock_excess_return, market_cap_on_current_day)

    daily_standard_deviation_exposure = get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day)

    cumulative_range_exposure = get_cumulative_range(stock_list, latest_trading_date, market_cap_on_current_day)

    historical_sigma_exposure = get_historical_sigma(stock_excess_return, market_portfolio_excess_return, market_portfolio_beta, market_portfolio_beta_exposure, market_cap_on_current_day)

    residual_volatility_exposure = 0.74 * daily_standard_deviation_exposure + 0.16 * cumulative_range_exposure + 0.1 * historical_sigma_exposure

    orthogonalized_residual_volatility_exposure = orthogonalize(target_variable=residual_volatility_exposure,reference_variable=market_portfolio_beta_exposure,regression_weight=np.sqrt(market_cap_on_current_day) / (np.sqrt(market_cap_on_current_day).sum()))

    residual_volatility_exposure = winsorization_and_market_cap_weighed_standardization(orthogonalized_residual_volatility_exposure, market_cap_on_current_day)

    momentum_exposure = get_momentum(stock_list, latest_trading_date, market_cap_on_current_day)

    liquidity_exposure = get_liquidity(stock_list, latest_trading_date, market_cap_on_current_day)

    earnings_to_price = get_earning_to_price_ratio(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    cash_earnings_to_price = get_cash_earnings_to_price_ratio(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    earnings_yield = earnings_to_price*(11/32) + cash_earnings_to_price*(21/32)

    earnings_yield = winsorization_and_market_cap_weighed_standardization(earnings_yield,market_cap_on_current_day)

    book_to_price = book_to_price_ratio(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    market_leverage = get_market_leverage(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    debt_to_asset = get_debt_to_asset(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    book_leverage = get_book_leverage(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    leverage = market_leverage*0.38+debt_to_asset*0.35+book_leverage*0.27

    leverage = winsorization_and_market_cap_weighed_standardization(leverage,market_cap_on_current_day)

    sales_growth = get_sales_growth(latest_trading_date.strftime('%Y-%m-%d'),year,market_cap_on_current_day)

    earnings_gorwth = get_earnings_growth(latest_trading_date.strftime('%Y-%m-%d'),year,market_cap_on_current_day)

    growth = sales_growth*(47/71)+earnings_gorwth*(24/71)

    growth = winsorization_and_market_cap_weighed_standardization(growth,market_cap_on_current_day)

    style_factors = pd.concat([size_exposure, non_linear_size_exposure, market_portfolio_beta_exposure, residual_volatility_exposure, momentum_exposure, liquidity_exposure,earnings_yield,book_to_price,leverage,growth], axis = 1)

    style_factors.columns = ['size', 'non_linear_size', 'beta', 'residual_volatility', 'momentum', 'liquidity','earnings_yield','book_to_price','leverage','growth']

    return style_factors
