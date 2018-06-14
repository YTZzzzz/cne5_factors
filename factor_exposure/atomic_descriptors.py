import sys

sys.path.append("/Users/rice/Documents/cne5_factors/factor_exposure/")

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


def get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life = 42, length = 252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    # 把股票超额收益率中的缺失值替换为 0（缺失值多于66个的股票已被剔除）

    sum_of_squares = (weighted_stock_excess_return - weighted_stock_excess_return.mean()).replace(np.nan,0).pow(2).sum()

    weighted_stock_standard_deviation = sum_of_squares.divide(len(stock_excess_return) - 1).pow(0.5)

    processed_weighted_stock_standard_deviation = winsorization_and_market_cap_weighed_standardization(weighted_stock_standard_deviation, market_cap_on_current_day)

    return processed_weighted_stock_standard_deviation


def get_cumulative_range(stock_list, date, market_cap_on_current_day):

    trading_date_253_before = rqdatac.get_trading_dates(date - timedelta(days=500), date, country='cn')[-253]

    daily_return = rqdatac.get_price(stock_list, trading_date_253_before, date, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据少于66个的股票

    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > (len(daily_return) - 66)].index

    daily_return = daily_return.drop(daily_return[inds], axis=1)

    # 把复利无风险日收益率转为日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_253_before, end_date=date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[daily_return.index]

    # 每21个交易日为一个时间区间

    spliting_points = np.arange(0, 273, 21)

    cummulative_return = pd.DataFrame()

    for period in range(1, len(spliting_points)):

        compounded_return = ((1 + daily_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        compounded_risk_free_return = ((1 + risk_free_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        cummulative_return[period] = np.log(1 + compounded_return).subtract(np.log(1 + compounded_risk_free_return.iloc[0]))

    processed_cumulative_range = winsorization_and_market_cap_weighed_standardization(cummulative_return.T.max() - cummulative_return.T.min(), market_cap_on_current_day)

    return processed_cumulative_range


def get_historical_sigma(stock_excess_return, market_portfolio_excess_return, market_portfolio_beta, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life = 63, length = 252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_residual_volatility = pd.Series()

    for stock in stock_excess_return.columns:

        alpha =  weighted_stock_excess_return[stock].mean() - market_portfolio_beta[stock] * weighted_market_portfolio_excess_return.mean()

        weighted_residual_volatility[stock] = (weighted_stock_excess_return[stock] - market_portfolio_beta[stock] * weighted_market_portfolio_excess_return - alpha).std()

    # 相对于贝塔正交化，降低波动率因子和贝塔因子的共线性

    processed_weighted_residual_volatility = winsorization_and_market_cap_weighed_standardization(weighted_residual_volatility, market_cap_on_current_day[weighted_residual_volatility.index])

    # 上述贝塔个阿尔法的计算，和 statsmodel 的计算结果对比

    #import statsmodels.api as st

    #X = np.stack([weighted_market_portfolio_excess_return.values, np.ones(len(weighted_market_portfolio_excess_return.values))]).T

    #Y = weighted_stock_excess_return['300364.XSHE'].values

    #st_result = st.OLS(Y, X).fit()

    #print(st_result.params)

    #print(weighted_market_portfolio_excess_return.cov(weighted_stock_excess_return['300364.XSHE']) / weighted_market_portfolio_excess_return.var())

    #alpha = weighted_stock_excess_return['300364.XSHE'].mean() - st_result.params[0] * weighted_market_portfolio_excess_return.mean()

    #print(alpha)

    return processed_weighted_residual_volatility


def get_trailing_earning_to_price_ratio(date, market_cap_on_current_day, recent_report_type, annual_report_type):

    net_profit_ttm = get_ttm_sum(rqdatac.financials.income_statement.net_profit, date, recent_report_type, annual_report_type)

    earning_to_price = (net_profit_ttm/market_cap_on_current_day[net_profit_ttm.index]).T

    processed_earning_to_price= winsorization_and_market_cap_weighed_standardization(earning_to_price, market_cap_on_current_day[earning_to_price.index])

    return processed_earning_to_price


# style:leverage

# MLEV: Market leverage = (ME+PE+LD)/ME ME:最新市值 PE:最新优先股账面价值 LD:长期负债账面价值

# 根据 Barra 因子解释：total debt=long term debt+current liabilities,在因子计算中使用非流动负债合计：non_current_liabilities作为long_term_debt

def get_market_leverage(market_cap_on_current_day, last_reported_non_current_liabilities, last_reported_preferred_stock):

    market_leverage = (market_cap_on_current_day + last_reported_non_current_liabilities + last_reported_preferred_stock)/market_cap_on_current_day

    processed_market_leverage = winsorization_and_market_cap_weighed_standardization(market_leverage, market_cap_on_current_day)

    return processed_market_leverage


# DTOA:Debt_to_asset：total debt/total assets

def get_debt_to_assets(market_cap_on_current_day, recent_report_type):

    total_debt = get_last_reported_values(rqdatac.financials.balance_sheet.total_liabilities, recent_report_type)

    total_asset = get_last_reported_values(rqdatac.financials.balance_sheet.total_assets, recent_report_type)

    debt_to_asset = total_debt/total_asset

    processed_debt_to_asset = winsorization_and_market_cap_weighed_standardization(debt_to_asset, market_cap_on_current_day)

    return processed_debt_to_asset


# BLEV: book leverage = (BE+PE+LD)/BE BE:普通股权账面价值 PE：优先股账面价值 LD:长期负债账面价值

# 普通股账面价值（book value of common equity，BE）用实收资本（paid in capital）表示

def get_book_leverage(market_cap_on_current_day, last_reported_non_current_liabilities, last_reported_preferred_stock, recent_report_type):

    book_value_of_common_stock = get_last_reported_values(rqdatac.financials.balance_sheet.paid_in_capital, recent_report_type)

    book_leverage = (book_value_of_common_stock + last_reported_preferred_stock + last_reported_non_current_liabilities)/book_value_of_common_stock

    processed_book_leverage = winsorization_and_market_cap_weighed_standardization(book_leverage, market_cap_on_current_day)

    return processed_book_leverage


def get_sales_growth(date, market_cap_on_current_day, recent_five_annual_shares, recent_report_type):

    recent_five_annual_sales_revenue = recent_five_annual_values(rqdatac.financials.income_statement.revenue, date, recent_report_type)

    # 上面函数默认取出当前所有上市股票的财务数据，包含一部分在计算日期未上市的股票，因此需要取子集

    recent_five_annual_sales_revenue = recent_five_annual_sales_revenue.loc[market_cap_on_current_day.index]

    sales_revenue_per_share = recent_five_annual_sales_revenue.divide(recent_five_annual_shares).dropna()

    x = np.array([5, 4, 3, 2, 1])

    # 经验方差和经验协方差均要除以（样本数 - 1），分子分母相除，该常数项约去

    variance_of_x = np.sum((x - x.mean()) ** 2)

    covariance_of_xy = (x - x.mean()).dot(sales_revenue_per_share.T - sales_revenue_per_share.T.mean(axis=0))

    regression_coefficient = pd.Series(data = covariance_of_xy/variance_of_x, index = sales_revenue_per_share.index)

    return regression_coefficient/abs(sales_revenue_per_share).T.mean()


def get_earnings_growth(date, market_cap_on_current_day, recent_five_annual_shares, recent_report_type):

    recent_five_annual_net_profit = recent_five_annual_values(rqdatac.financials.income_statement.net_profit, date, recent_report_type)

    # 上面函数默认取出当前所有上市股票的财务数据，包含一部分在计算日期未上市的股票，因此需要取子集

    recent_five_annual_net_profit = recent_five_annual_net_profit.loc[market_cap_on_current_day.index]

    earnings_per_share = recent_five_annual_net_profit.divide(recent_five_annual_shares).dropna()

    x = np.array([5, 4, 3, 2, 1])

    # 经验方差和经验协方差均要除以（样本数 - 1），分子分母相除，该常数项约去

    variance_of_x = np.sum((x - x.mean()) ** 2)

    covariance_of_xy = (x - x.mean()).dot(earnings_per_share.T - earnings_per_share.T.mean(axis=0))

    regression_coefficient = pd.Series(data = covariance_of_xy/variance_of_x, index = earnings_per_share.index)

    return regression_coefficient/abs(earnings_per_share).T.mean()




