
from intermediate_variables import *
from operators import *

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


import rqdatac
rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))


def get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life=63, length=252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_market_portfolio_variance = weighted_market_portfolio_excess_return.var()

    beta = [weighted_market_portfolio_excess_return.cov(
        weighted_stock_excess_return[stock]) / weighted_market_portfolio_variance for stock in
            stock_excess_return.columns]

    market_portfolio_beta = pd.Series(beta, index=stock_excess_return.columns)

    market_portfolio_beta_exposure = winsorization_and_market_cap_weighed_standardization(market_portfolio_beta,
                                                                                          market_cap_on_current_day)

    # 细分因子 historical_sigma 的计算需要 beta 的原始值，所以同时返回原始暴露度和标准化暴露度

    return market_portfolio_beta, market_portfolio_beta_exposure


def get_momentum(stock_list, date, market_cap_on_current_day):

    trading_date_525_before = rqdatac.get_trading_dates(date - timedelta(days=1000), date, country='cn')[-525]

    trading_date_21_before = rqdatac.get_trading_dates(date - timedelta(days=40), date, country='cn')[-21]

    # 共需要 525 - 21 = 504 个交易日的收益率

    exp_weight = get_exponential_weight(half_life=126, length=504)

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    daily_return = rqdatac.get_price(stock_list, trading_date_525_before, trading_date_21_before, frequency='1d',
                                     fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据存在空值的股票

    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > 0].index

    daily_return = daily_return.drop(daily_return[inds], axis=1)

    # 把复利无风险日收益率转为日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_525_before, end_date=date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[daily_return.index]

    relative_strength = np.log(1 + daily_return).T.subtract(np.log(1 + risk_free_return.iloc[:, 0])).dot(exp_weight)

    processed_relative_strength = winsorization_and_market_cap_weighed_standardization(relative_strength,
                                                                                       market_cap_on_current_day[
                                                                                           relative_strength.index])

    return processed_relative_strength


# 计算residual volatility
def get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life = 42, length = 252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    sum_of_squares = (weighted_stock_excess_return - weighted_stock_excess_return.mean()).pow(2).sum()

    weighted_stock_standard_deviation = sum_of_squares.divide(len(stock_excess_return) - 1).pow(0.5)

    processed_weighted_stock_standard_deviation = winsorization_and_market_cap_weighed_standardization(weighted_stock_standard_deviation, market_cap_on_current_day)

    return processed_weighted_stock_standard_deviation


def get_cumulative_range(stock_list, date, market_cap_on_current_day):

    trading_date_253_before = rqdatac.get_trading_dates(date - timedelta(days=500), date, country='cn')[-253]

    daily_return = rqdatac.get_price(stock_list, trading_date_253_before, date, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据存在空值的股票

    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > 0].index

    daily_return = daily_return.drop(daily_return[inds], axis=1)

    # 把复利无风险日收益率转为日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_253_before, end_date=date, tenor='3M')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[daily_return.index]

    # 每21个交易日为一个时间区间

    spliting_points = np.arange(0, 273, 21)

    cummulative_return = pd.DataFrame()

    for period in range(1, len(spliting_points)):

        compounded_return = ((1 + daily_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        compounded_risk_free_return = ((1 + risk_free_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        cummulative_return[period] = np.log(1 + compounded_return).subtract(np.log(1 + compounded_risk_free_return.iloc[0]))

    cummulative_return = cummulative_return.cumsum(axis=1)

    processed_cumulative_range = winsorization_and_market_cap_weighed_standardization(cummulative_return.T.max() - cummulative_return.T.min(), market_cap_on_current_day)

    return processed_cumulative_range


def get_historical_sigma(stock_excess_return, market_portfolio_excess_return, market_portfolio_beta,market_portfolio_beta_exposure, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life = 63, length = 252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_residual_volatility = pd.Series()

    for stock in stock_excess_return.columns:

        alpha =  weighted_stock_excess_return[stock].mean() - market_portfolio_beta[stock] * weighted_market_portfolio_excess_return.mean()

        weighted_residual_volatility[stock] = (weighted_stock_excess_return[stock] - market_portfolio_beta[stock] * weighted_market_portfolio_excess_return - alpha).std()

    # 相对于贝塔正交化，降低波动率因子和贝塔因子的共线性

    orthogonalized_weighted_residual_volatility = orthogonalize(target_variable = weighted_residual_volatility, reference_variable = market_portfolio_beta_exposure, regression_weight = np.sqrt(market_cap_on_current_day)/(np.sqrt(market_cap_on_current_day).sum()))

    processed_weighted_residual_volatility = winsorization_and_market_cap_weighed_standardization(orthogonalized_weighted_residual_volatility, market_cap_on_current_day[weighted_residual_volatility.index])

    return processed_weighted_residual_volatility


def get_residual_volatility(stock_list, latest_trading_date, stock_excess_return, market_portfolio_excess_return,market_cap_on_current_day, market_portfolio_beta_exposure, market_portfolio_beta):

    daily_standard_deviation_exposure = get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day)

    cumulative_range_exposure = get_cumulative_range(stock_list, latest_trading_date, market_cap_on_current_day)

    historical_sigma_exposure = get_historical_sigma(stock_excess_return, market_portfolio_excess_return,market_portfolio_beta, market_portfolio_beta_exposure,market_cap_on_current_day)

    atomic_descriptors_df = pd.concat([daily_standard_deviation_exposure, cumulative_range_exposure, historical_sigma_exposure], axis=1)

    atomic_descriptors_df.columns = ['daily_standard_deviation', 'cumulative_range', 'historical_sigma']

    atom_descriptors_weight = pd.Series(data=[0.74, 0.16, 0.1],index=['daily_standard_deviation', 'cumulative_range', 'historical_sigma'])

    residual_volatility = atomic_descriptors_imputation_and_combination(atomic_descriptors_df, atom_descriptors_weight)

    orthogonalized_weighted_residual_volatility = orthogonalize(target_variable=residual_volatility,reference_variable=market_portfolio_beta_exposure,regression_weight=np.sqrt(market_cap_on_current_day) / (np.sqrt(market_cap_on_current_day).sum()))

    processed_residual_volatility_exposure = winsorization_and_market_cap_weighed_standardization(orthogonalized_weighted_residual_volatility, market_cap_on_current_day)

    return daily_standard_deviation_exposure, cumulative_range_exposure, historical_sigma_exposure, processed_residual_volatility_exposure


def get_momentum_and_res_vol(date):

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_252_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-252]

    stock_list = rqdatac.all_instruments(type='CS', date=latest_trading_date)['order_book_id'].values.tolist()

    ### 获取因子计算共用的行情数据和财务数据

    recent_report_type, annual_report_type, market_cap_on_current_day, \
    stock_excess_return, market_portfolio_excess_return, recent_five_annual_shares, \
    last_reported_non_current_liabilities, last_reported_preferred_stock = get_financial_and_market_data(stock_list,latest_trading_date,trading_date_252_before)

    # momentum和residual volatility计算

    market_portfolio_beta, market_portfolio_beta_exposure = get_market_portfolio_beta(stock_excess_return,market_portfolio_excess_return,market_cap_on_current_day)

    momentum = get_momentum(stock_list, latest_trading_date, market_cap_on_current_day)

    daily_standard_deviation, cumulative_range, historical_sigma, residual_volatility = get_residual_volatility(
        stock_list, latest_trading_date, stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day,
        market_portfolio_beta_exposure, market_portfolio_beta)

    style_factors_exposure = pd.concat([momentum, residual_volatility], axis=1)

    style_factors_exposure.columns = ['momentum', 'residual_volatility']

    atomic_descriptors_exposure = pd.concat(
        [daily_standard_deviation, cumulative_range, historical_sigma], axis=1)

    atomic_descriptors_exposure.columns = ['daily_standard_deviation', 'cumulative_range', 'historical_sigma']

    # 提取财务数据的时候，会提取当前全市场股票的数据，因此 dataframe 中可能包含计算日期未上市的股票，需要对 style_factors_exposure 取子集

    atomic_descriptors_exposure = atomic_descriptors_exposure.loc[stock_list]

    style_factors_exposure = style_factors_exposure.loc[stock_list]

    # 用回归方法处理细分因子的缺失值

    imputed_atomic_descriptors = pd.DataFrame()

    for atomic_descriptor in atomic_descriptors_exposure.columns:
        imputed_atomic_descriptors[atomic_descriptor] = individual_factor_imputation(stock_list, atomic_descriptors_exposure[atomic_descriptor], market_cap_on_current_day,latest_trading_date.strftime('%Y-%m-%d'))

    # 用回归方法处理风格因子暴露度的缺失值

    imputed_style_factors_exposure = style_factors_imputation(style_factors_exposure, market_cap_on_current_day,latest_trading_date.strftime('%Y-%m-%d'))

    # 若经过缺失值处理后因子暴露度依旧存在缺失值，使用全市场股票进行回归，填补缺失值

    if imputed_style_factors_exposure.isnull().sum().sum() > 0:

        imputed_style_factors_exposure = factor_imputation(market_cap_on_current_day,imputed_style_factors_exposure)

    if imputed_atomic_descriptors.isnull().sum().sum() > 0:

        imputed_atomic_descriptors = factor_imputation(market_cap_on_current_day,imputed_atomic_descriptors)

    return imputed_atomic_descriptors, imputed_style_factors_exposure