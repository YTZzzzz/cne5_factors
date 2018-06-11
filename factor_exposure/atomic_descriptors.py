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


def get_size(market_cap_on_current_day):

    processed_size = winsorization_and_market_cap_weighed_standardization(np.log(market_cap_on_current_day.replace(0, np.nan)), market_cap_on_current_day)

    return processed_size


def get_non_linear_size(size_exposure, market_cap_on_current_day):

    cubed_size = np.power(size_exposure, 3)

    processed_cubed_size = winsorization_and_market_cap_weighed_standardization(cubed_size, market_cap_on_current_day)

    orthogonalized_cubed_size = orthogonalize(target_variable = processed_cubed_size, reference_variable = size_exposure, regression_weight = np.sqrt(market_cap_on_current_day)/(np.sqrt(market_cap_on_current_day).sum()))

    processed_orthogonalized_cubed_size = winsorization_and_market_cap_weighed_standardization(orthogonalized_cubed_size, market_cap_on_current_day)

    return processed_orthogonalized_cubed_size


def get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life = 63, length = 252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_market_portfolio_variance = weighted_market_portfolio_excess_return.var()

    beta = [weighted_market_portfolio_excess_return.cov(weighted_stock_excess_return[stock])/weighted_market_portfolio_variance for stock in stock_excess_return.columns]

    market_portfolio_beta = pd.Series(beta, index = stock_excess_return.columns)

    market_portfolio_beta_exposure = winsorization_and_market_cap_weighed_standardization(market_portfolio_beta, market_cap_on_current_day)

    # 细分因子 historical_sigma 的计算需要 beta 的原始值，所以同时返回原始暴露度和标准化暴露度

    return market_portfolio_beta, market_portfolio_beta_exposure


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

        weighted_residual_volatility[stock] = ((weighted_stock_excess_return[stock] - market_portfolio_beta[stock] * weighted_market_portfolio_excess_return - alpha).std()

    # 相对于贝塔正交化，降低波动率因子和贝塔因子的共线性

    processed_weighted_residual_volatiltiy = winsorization_and_market_cap_weighed_standardization(weighted_residual_volatiltiy, market_cap_on_current_day[weighted_residual_volatiltiy.index])

    return processed_weighted_residual_volatiltiy


def get_momentum(stock_list, date, market_cap_on_current_day):

    trading_date_505_before = rqdatac.get_trading_dates(date - timedelta(days=1000), date, country='cn')[-504]

    trading_date_21_before = rqdatac.get_trading_dates(date - timedelta(days=40), date, country='cn')[-21]

    # 共需要 504 - 21 = 483 个交易日的收益率

    exp_weight = get_exponential_weight(half_life = 126, length = 483)

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    daily_return = rqdatac.get_price(stock_list, trading_date_505_before, trading_date_21_before, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据少于66个的股票

    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > (len(daily_return) - 66)].index

    daily_return = daily_return.drop(daily_return[inds], axis=1)

    # 把复利无风险日收益率转为日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_505_before, end_date=date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[daily_return.index]

    relative_strength = np.log(1 + daily_return).T.subtract(np.log(1 + risk_free_return.iloc[:, 0])).dot(exp_weight)

    processed_relative_strength =  winsorization_and_market_cap_weighed_standardization(relative_strength, market_cap_on_current_day[relative_strength.index])

    return processed_relative_strength


def get_liquidity(stock_list, date, market_cap_on_current_day):

    trading_date_252_before = rqdatac.get_trading_dates(date - timedelta(days=500), date, country='cn')[-252]

    trading_volume = rqdatac.get_price(stock_list, trading_date_252_before, date, frequency='1d', fields='volume')

    outstanding_shares = rqdatac.get_shares(stock_list, trading_date_252_before, date, fields='total_a')

    # 成交量加1避免长期停牌导致的换手率为0，导致取对数时报错的问题

    daily_turnover_rate = (trading_volume + 1).divide(outstanding_shares)

    # 对于对应时期内换手率为 0 的股票，其细分因子暴露度也设为0

    one_month_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-21:].sum()), market_cap_on_current_day)

    three_months_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-63:].sum()/3), market_cap_on_current_day)

    twelve_months_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-252:].sum()/12), market_cap_on_current_day)

    liquidity = 0.35*one_month_share_turnover + 0.35*three_months_share_turnover + 0.3*twelve_months_share_turnover

    processed_liquidity =  winsorization_and_market_cap_weighed_standardization(liquidity, market_cap_on_current_day)

    return processed_liquidity


def get_trailing_earning_to_price_ratio(date, market_cap_on_current_day, recent_report_type, annual_report_type):

    net_profit_ttm = get_ttm_sum(rqdatac.financials.income_statement.net_profit, date, recent_report_type, annual_report_type)

    earning_to_price = (net_profit_ttm/market_cap_on_current_day[net_profit_ttm.index]).T

    processed_earning_to_price= winsorization_and_market_cap_weighed_standardization(earning_to_price, market_cap_on_current_day[earning_to_price.index])

    return processed_earning_to_price


# CETOP:Trailing cash earning to price ratio

def get_cash_earnings_to_price_ratio(date, market_cap_on_current_day, recent_report_type, annual_report_type):

    cash_earnings = get_ttm_sum(rqdatac.financials.financial_indicator.earnings_per_share, date, recent_report_type, annual_report_type)

    stock_list = cash_earnings.index.tolist()

    share_price = rqdatac.get_price(stock_list, start_date=date, end_date=date, fields='close', adjust_type='pre')

    cash_earnings_to_price = cash_earnings / share_price.T[date]

    processed_cash_earning = winsorization_and_market_cap_weighed_standardization(cash_earnings_to_price, market_cap_on_current_day)

    return processed_cash_earning


# book-to-price = (股东权益合计-优先股)/市值

def get_book_to_price_ratio(date, market_cap_on_current_day, last_reported_preferred_stock, recent_report_type):

    total_equity = get_last_reported_values(rqdatac.financials.balance_sheet.total_equity, date, recent_report_type)

    book_to_price_ratio = (total_equity - last_reported_preferred_stock)/market_cap_on_current_day[total_equity.index]

    processed_book_to_price_ratio = winsorization_and_market_cap_weighed_standardization(book_to_price_ratio, market_cap_on_current_day[total_equity.index])

    return processed_book_to_price_ratio


# style:leverage

# MLEV: Market leverage = (ME+PE+LD)/ME ME:最新市值 PE:最新优先股账面价值 LD:长期负债账面价值

# 根据 Barra 因子解释：total debt=long term debt+current liabilities,在因子计算中使用非流动负债合计：non_current_liabilities作为long_term_debt

def get_market_leverage(market_cap_on_current_day, last_reported_non_current_liabilities, last_reported_preferred_stock):

    market_leverage = (market_cap_on_current_day + last_reported_non_current_liabilities + last_reported_preferred_stock)/market_cap_on_current_day

    processed_market_leverage = winsorization_and_market_cap_weighed_standardization(market_leverage, market_cap_on_current_day)

    return processed_market_leverage


# DTOA:Debt_to_asset：total debt/total assets

def get_debt_to_asset(date, market_cap_on_current_day, recent_report_type):

    total_debt = get_last_reported_values(rqdatac.financials.balance_sheet.total_liabilities, date, recent_report_type)

    total_asset = get_last_reported_values(rqdatac.financials.balance_sheet.total_assets, date, recent_report_type)

    debt_to_asset = total_debt/total_asset

    processed_debt_to_asset = winsorization_and_market_cap_weighed_standardization(debt_to_asset, market_cap_on_current_day)

    return processed_debt_to_asset


# BLEV: book leverage = (BE+PE+LD)/BE BE:普通股权账面价值 PE：优先股账面价值 LD:长期负债账面价值

# 普通股账面价值（book value of common equity，BE）用实收资本（paid in capital）表示

def get_book_leverage(date, market_cap_on_current_day, last_reported_non_current_liabilities, last_reported_preferred_stock, recent_report_type):

    book_value_of_common_stock = get_last_reported_values(rqdatac.financials.balance_sheet.paid_in_capital, date, recent_report_type)

    book_leverage = (book_value_of_common_stock + last_reported_preferred_stock + last_reported_non_current_liabilities)/book_value_of_common_stock

    processed_book_leverage = winsorization_and_market_cap_weighed_standardization(book_leverage, market_cap_on_current_day)

    return processed_book_leverage


def get_sales_growth(date, market_cap_on_current_day, recent_five_annual_shares):

    recent_five_annual_sales_revenue = recent_five_annual_values(rqdatac.financials.income_statement.revenue, date)

    # 上面函数默认取出当前所有上市股票的财务数据，包含一部分在计算日期未上市的股票，因此需要取子集

    recent_five_annual_sales_revenue = recent_five_annual_sales_revenue.loc[market_cap_on_current_day.index]

    sales_revenue_per_share = recent_five_annual_sales_revenue.divide(recent_five_annual_shares).dropna()

    x = np.array([5, 4, 3, 2, 1])

    # 经验方差和经验协方差均要除以（样本数 - 1），分子分母相除，该常数项约去

    variance_of_x = np.sum((x - x.mean()) ** 2)

    covariance_of_xy = (x - x.mean()).dot(sales_revenue_per_share.T - sales_revenue_per_share.T.mean(axis=0))

    regression_coefficient = pd.Series(data = covariance_of_xy/variance_of_x, index = sales_revenue_per_share.index)


    # 对比上述最小二乘法计算和 statsmodel 的计算结果

    #import statsmodels.api as st

    #X = np.stack([x, np.ones(len(x))]).T

    #Y = sales_revenue_per_share.loc['603993.XSHG']

    #st_result = st.OLS(Y.values, X).fit()

    #print(regression_coefficient.loc['603993.XSHG'])

    #print(st_result.params[0])


    return regression_coefficient/sales_revenue_per_share.T.mean()



def get_earnings_growth(date, market_cap_on_current_day, recent_five_annual_shares):

    recent_five_annual_net_profit = recent_five_annual_values(rqdatac.financials.income_statement.net_profit, date)

    # 上面函数默认取出当前所有上市股票的财务数据，包含一部分在计算日期未上市的股票，因此需要取子集

    recent_five_annual_net_profit = recent_five_annual_net_profit.loc[market_cap_on_current_day.index]

    earnings_per_share = recent_five_annual_net_profit.divide(recent_five_annual_shares).dropna()

    x = np.array([5, 4, 3, 2, 1])

    # 经验方差和经验协方差均要除以（样本数 - 1），分子分母相除，该常数项约去

    variance_of_x = np.sum((x - x.mean()) ** 2)

    covariance_of_xy = (x - x.mean()).dot(earnings_per_share.T - earnings_per_share.T.mean(axis=0))

    regression_coefficient = pd.Series(data = covariance_of_xy/variance_of_x, index = earnings_per_share.index)

    return regression_coefficient/earnings_per_share.T.mean()





date = '2018-02-06'

year = pd.Series([48, 36, 24, 12, 0])


def get_style_factors(date):

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_252_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-252]

    stock_list = rqdatac.all_instruments(type = 'CS', date = latest_trading_date)['order_book_id'].values.tolist()




    ### 细分因子计算

    size_exposure = size(market_cap_on_current_day)

    non_linear_size_exposure = non_linear_size(size_exposure, market_cap_on_current_day)

    market_portfolio_beta, market_portfolio_beta_exposure = get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day)

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


