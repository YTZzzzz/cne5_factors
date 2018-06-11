import sys

sys.path.append("/Users/jjj728/git/cne5_factors/factor_exposure/")

from intermediate_variables import *
from operators import *

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import rqdatac

rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))


def size(market_cap_on_current_day):
    processed_size = winsorization_and_market_cap_weighed_standardization(
        np.log(market_cap_on_current_day.replace(0, np.nan)), market_cap_on_current_day)

    return processed_size


def non_linear_size(size_exposure, market_cap_on_current_day):
    cubed_size = np.power(size_exposure, 3)

    processed_cubed_size = winsorization_and_market_cap_weighed_standardization(cubed_size, market_cap_on_current_day)

    orthogonalized_cubed_size = orthogonalize(target_variable=processed_cubed_size, reference_variable=size_exposure,
                                              regression_weight=np.sqrt(market_cap_on_current_day) / (
                                                  np.sqrt(market_cap_on_current_day).sum()))

    processed_orthogonalized_cubed_size = winsorization_and_market_cap_weighed_standardization(
        orthogonalized_cubed_size, market_cap_on_current_day)

    return processed_orthogonalized_cubed_size


def get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day):
    benchmark_variance = market_portfolio_excess_return.var()

    beta = [market_portfolio_excess_return.cov(stock_excess_return[stock]) / benchmark_variance for stock in
            stock_excess_return.columns]

    # 不考虑基准组合的贝塔

    market_portfolio_beta = pd.Series(beta, index=stock_excess_return.columns)

    processed_market_portfolio_beta = winsorization_and_market_cap_weighed_standardization(market_portfolio_beta,
                                                                                           market_cap_on_current_day)

    # 细分因子 historical_sigma 的计算需要 beta 的原始值，所以同时返回原始暴露度和标准化暴露度

    return market_portfolio_beta, processed_market_portfolio_beta


def get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day):
    exp_weight = get_exponential_weight(half_life=42, length=252)

    # 把股票超额收益率中的缺失值替换为 0（缺失值多于66个的股票已被剔除）

    weighted_stock_standard_deviation = (stock_excess_return.replace(np.nan, 0) - stock_excess_return.mean()).pow(
        2).T.dot(exp_weight).pow(0.5)

    processed_weighted_stock_standard_deviation = winsorization_and_market_cap_weighed_standardization(
        weighted_stock_standard_deviation, market_cap_on_current_day)

    return processed_weighted_stock_standard_deviation


def get_cumulative_range(stock_list, date, market_cap_on_current_day):
    trading_date_253_before = rqdatac.get_trading_dates(date - timedelta(days=500), date, country='cn')[-253]

    daily_return = rqdatac.get_price(stock_list, trading_date_253_before, date, frequency='1d',
                                     fields='ClosingPx').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据少于66个的股票

    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > (len(daily_return) - 66)].index

    daily_return = daily_return.drop(daily_return[inds], axis=1)

    # 把复利无风险日收益率转为日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_253_before, end_date=date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1).loc[daily_return.index]

    # 每21个交易日为一个时间区间

    spliting_points = np.arange(0, 273, 21)

    cummulative_return = pd.DataFrame()

    for period in range(1, len(spliting_points)):
        compounded_return = ((1 + daily_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        compounded_risk_free_return = \
        ((1 + risk_free_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        cummulative_return[period] = np.log(1 + compounded_return).subtract(
            np.log(1 + compounded_risk_free_return.iloc[0]))

    processed_cumulative_range = winsorization_and_market_cap_weighed_standardization(
        cummulative_return.T.max() - cummulative_return.T.min(), market_cap_on_current_day)

    return processed_cumulative_range


def get_historical_sigma(stock_excess_return, market_portfolio_excess_return, market_portfolio_beta,
                         market_portfolio_beta_exposure, market_cap_on_current_day):
    exp_weight = get_exponential_weight(half_life=63, length=252)

    weighted_residual_volatiltiy = pd.Series()

    for stock in stock_excess_return.columns:
        weighted_residual_volatiltiy[stock] = (
            (stock_excess_return[stock] - market_portfolio_beta[stock] * market_portfolio_excess_return).replace(np.nan,
                                                                                                                 0).multiply(
                exp_weight)).std()

    # 相对于贝塔正交化，降低波动率因子和贝塔因子的共线性

    processed_weighted_residual_volatiltiy = winsorization_and_market_cap_weighed_standardization(
        weighted_residual_volatiltiy, market_cap_on_current_day)

    orthogonalized_weighted_residual_volatility = orthogonalize(target_variable=processed_weighted_residual_volatiltiy,
                                                                reference_variable=market_portfolio_beta_exposure,
                                                                regression_weight=np.sqrt(market_cap_on_current_day) / (
                                                                    np.sqrt(market_cap_on_current_day).sum()))

    processed_orthogonalized_weighted_residual_volatility = winsorization_and_market_cap_weighed_standardization(
        orthogonalized_weighted_residual_volatility, market_cap_on_current_day)

    return processed_orthogonalized_weighted_residual_volatility


def get_momentum(stock_list, date, market_cap_on_current_day):
    trading_date_505_before = rqdatac.get_trading_dates(date - timedelta(days=1000), date, country='cn')[-504]

    trading_date_21_before = rqdatac.get_trading_dates(date - timedelta(days=40), date, country='cn')[-21]

    # 共需要 504 - 21 = 483 个交易日的收益率

    exp_weight = get_exponential_weight(half_life=126, length=483)

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    daily_return = rqdatac.get_price(stock_list, trading_date_505_before, trading_date_21_before, frequency='1d',
                                     fields='ClosingPx').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据少于66个的股票

    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > (len(daily_return) - 66)].index

    daily_return = daily_return.drop(daily_return[inds], axis=1)

    # 把复利无风险日收益率转为日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_505_before, end_date=date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1).loc[daily_return.index]

    relative_strength = np.log(1 + daily_return).T.subtract(np.log(1 + risk_free_return.iloc[:, 0])).dot(exp_weight)

    processed_relative_strength = winsorization_and_market_cap_weighed_standardization(relative_strength,
                                                                                       market_cap_on_current_day)

    return processed_relative_strength


def get_liquidity(stock_list, date, market_cap_on_current_day):
    trading_date_252_before = rqdatac.get_trading_dates(date - timedelta(days=500), date, country='cn')[-252]

    trading_volume = rqdatac.get_price(stock_list, trading_date_252_before, date, frequency='1d', fields='volume')

    outstanding_shares = rqdatac.get_shares(stock_list, trading_date_252_before, date, fields='circulation_a')

    daily_turnover_rate = trading_volume.divide(outstanding_shares)

    # 对于对应时期内换手率为 0 的股票，其细分因子暴露度也设为0

    one_month_share_turnover = np.log(daily_turnover_rate.iloc[-21:].sum().replace(0, np.nan))

    three_months_share_turnover = np.log(daily_turnover_rate.iloc[-63:].sum().replace(0, np.nan) / 3)

    twelve_months_share_turnover = np.log(daily_turnover_rate.iloc[-252:].sum().replace(0, np.nan) / 12)

    liquidity = 0.35 * one_month_share_turnover.replace(np.nan, 0) + 0.35 * three_months_share_turnover.replace(np.nan,
                                                                                                                0) + 0.3 * twelve_months_share_turnover.replace(
        np.nan, 0)

    processed_liquidity = winsorization_and_market_cap_weighed_standardization(liquidity, market_cap_on_current_day)

    return processed_liquidity


date = '2018-02-02'


def get_style_factors(date):
    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_252_before = \
    rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-252]

    stock_list = rqdatac.all_instruments(type='CS', date=latest_trading_date)['order_book_id'].values.tolist()

    stock_excess_return, market_portfolio_excess_return = get_daily_excess_return(stock_list,
                                                                                  trading_date_252_before.strftime(
                                                                                      '%Y-%m-%d'),
                                                                                  latest_trading_date.strftime(
                                                                                      '%Y-%m-%d'))

    market_cap_on_current_day = rqdatac.get_factor(id_or_symbols=stock_excess_return.columns.tolist(),
                                                   factor='market_cap',
                                                   start_date=latest_trading_date.strftime('%Y-%m-%d'),
                                                   end_date=latest_trading_date.strftime('%Y-%m-%d'))

    size_exposure = size(market_cap_on_current_day)

    non_linear_size_exposure = non_linear_size(size_exposure, market_cap_on_current_day)

    market_portfolio_beta, market_portfolio_beta_exposure = get_market_portfolio_beta(stock_excess_return,
                                                                                      market_portfolio_excess_return,
                                                                                      market_cap_on_current_day)

    daily_standard_deviation_exposure = get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day)

    cumulative_range_exposure = get_cumulative_range(stock_list, latest_trading_date, market_cap_on_current_day)

    historical_sigma_exposure = get_historical_sigma(stock_excess_return, market_portfolio_excess_return,
                                                     market_portfolio_beta, market_portfolio_beta_exposure,
                                                     market_cap_on_current_day)

    residual_volatility_exposure = 0.74 * daily_standard_deviation_exposure + 0.16 * cumulative_range_exposure + 0.1 * historical_sigma_exposure

    momentum_exposure = get_momentum(stock_list, latest_trading_date, market_cap_on_current_day)

    liquidity_exposure = get_liquidity(stock_list, latest_trading_date, market_cap_on_current_day)

    style_factors = pd.concat(
        [size_exposure, non_linear_size_exposure, market_portfolio_beta_exposure, residual_volatility_exposure,
         momentum_exposure, liquidity_exposure], axis=1)

    style_factors.columns = ['size', 'non_linear_size', 'beta', 'residual_volatility', 'momentum', 'liquidity']

    return style_factors





