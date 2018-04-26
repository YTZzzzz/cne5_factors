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

rqdatac.init('ricequant', '8ricequant8',('q-tools.ricequant.com', 16010))


def size(market_cap_on_current_day):

    processed_size = winsorization_and_market_cap_weighed_standardization(np.log(market_cap_on_current_day.replace(0, np.nan)), market_cap_on_current_day)

    return processed_size


def non_linear_size(size_exposure, market_cap_on_current_day):

    cubed_size = np.power(size_exposure, 3)

    processed_cubed_size = winsorization_and_market_cap_weighed_standardization(cubed_size, market_cap_on_current_day)

    orthogonalized_cubed_size = orthogonalize(target_variable = processed_cubed_size, reference_variable = size_exposure, regression_weight = np.sqrt(market_cap_on_current_day)/(np.sqrt(market_cap_on_current_day).sum()))

    processed_orthogonalized_cubed_size = winsorization_and_market_cap_weighed_standardization(orthogonalized_cubed_size, market_cap_on_current_day)

    return processed_orthogonalized_cubed_size


def get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day):

    benchmark_variance = market_portfolio_excess_return.var()

    beta = [market_portfolio_excess_return.cov(stock_excess_return[stock])/benchmark_variance for stock in stock_excess_return.columns]

    # 不考虑基准组合的贝塔

    market_portfolio_beta = pd.Series(beta, index = stock_excess_return.columns)

    processed_market_portfolio_beta = winsorization_and_market_cap_weighed_standardization(market_portfolio_beta, market_cap_on_current_day)

    # 细分因子 historical_sigma 的计算需要 beta 的原始值，所以同时返回原始暴露度和标准化暴露度

    return market_portfolio_beta, processed_market_portfolio_beta


def get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life = 42, length = 252)

    # 把股票超额收益率中的缺失值替换为 0（缺失值多于66个的股票已被剔除）

    weighted_stock_standard_deviation = (stock_excess_return - stock_excess_return.mean()).replace(np.nan,0).pow(2).T.dot(exp_weight).pow(0.5)

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

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1).loc[daily_return.index]

    # 每21个交易日为一个时间区间

    spliting_points = np.arange(0, 273, 21)

    cummulative_return = pd.DataFrame()

    for period in range(1, len(spliting_points)):

        compounded_return = ((1 + daily_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        compounded_risk_free_return = ((1 + risk_free_return.iloc[spliting_points[0]:spliting_points[period]]).cumprod() - 1).iloc[-1]

        cummulative_return[period] = np.log(1 + compounded_return).subtract(np.log(1 + compounded_risk_free_return.iloc[0]))

    processed_cumulative_range = winsorization_and_market_cap_weighed_standardization(cummulative_return.T.max() - cummulative_return.T.min(), market_cap_on_current_day)

    return processed_cumulative_range


def get_historical_sigma(stock_excess_return, market_portfolio_excess_return, market_portfolio_beta, market_portfolio_beta_exposure, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life = 63, length = 252)

    weighted_residual_volatiltiy = pd.Series()

    for stock in stock_excess_return.columns:

        weighted_residual_volatiltiy[stock] = ((stock_excess_return[stock] - market_portfolio_beta[stock] * market_portfolio_excess_return).multiply(exp_weight)).std()

        #weighted_residual_volatiltiy[stock] = ((stock_excess_return[stock] - market_portfolio_beta[stock] * market_portfolio_excess_return).replace(np.nan, 0).multiply(exp_weight)).std()

    # 相对于贝塔正交化，降低波动率因子和贝塔因子的共线性

    processed_weighted_residual_volatiltiy = winsorization_and_market_cap_weighed_standardization(weighted_residual_volatiltiy, market_cap_on_current_day)

    orthogonalized_weighted_residual_volatility = orthogonalize(target_variable = processed_weighted_residual_volatiltiy, reference_variable = market_portfolio_beta_exposure, regression_weight = np.sqrt(market_cap_on_current_day)/(np.sqrt(market_cap_on_current_day).sum()))

    processed_orthogonalized_weighted_residual_volatility =  winsorization_and_market_cap_weighed_standardization(orthogonalized_weighted_residual_volatility, market_cap_on_current_day)

    return processed_orthogonalized_weighted_residual_volatility


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

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1).loc[daily_return.index]

    relative_strength = np.log(1 + daily_return).T.subtract(np.log(1 + risk_free_return.iloc[:, 0])).dot(exp_weight)

    processed_relative_strength =  winsorization_and_market_cap_weighed_standardization(relative_strength, market_cap_on_current_day)

    return processed_relative_strength


def get_liquidity(stock_list, date, market_cap_on_current_day):

    trading_date_252_before = rqdatac.get_trading_dates(date - timedelta(days=500), date, country='cn')[-252]

    stock_without_suspended_stock = drop_suspended_stock(stock_list,date)

    trading_volume = rqdatac.get_price(stock_without_suspended_stock, trading_date_252_before, date, frequency='1d', fields='volume')

    outstanding_shares = rqdatac.get_shares(stock_without_suspended_stock, trading_date_252_before, date, fields='total_a')

    daily_turnover_rate = trading_volume.divide(outstanding_shares)

    # 对于对应时期内换手率为 0 的股票，其细分因子暴露度也设为0

    one_month_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-21:].sum().replace(0, np.nan)),market_cap_on_current_day)

    three_months_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-63:].sum().replace(0, np.nan)/3),market_cap_on_current_day)

    twelve_months_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-252:].sum().replace(0, np.nan)/12),market_cap_on_current_day)

    liquidity = 0.35*one_month_share_turnover.replace(np.nan, 0) + 0.35*three_months_share_turnover.replace(np.nan, 0) + 0.3*twelve_months_share_turnover.replace(np.nan, 0)

    processed_liquidity =  winsorization_and_market_cap_weighed_standardization(liquidity, market_cap_on_current_day)

    return processed_liquidity


def get_earning_to_price_ratio(date,market_cap_on_current_day):

    net_profit_ttm = ttm_sum(rqdatac.financials.income_statement.net_profit,date)

    ep_ratio = (net_profit_ttm/market_cap_on_current_day[net_profit_ttm.index]).T

    processed_ep =  winsorization_and_market_cap_weighed_standardization(ep_ratio, market_cap_on_current_day)

    return processed_ep


# CETOP:Trailing cash earning to price ratio
def get_cash_earnings_to_price_ratio(date,market_cap_on_current_day):

    cash_flow_from_operating_activities_ttm = ttm_sum(rqdatac.financials.cash_flow_statement.cash_flow_from_operating_activities,date)

    stock_list = cash_flow_from_operating_activities_ttm.index.tolist()

    total_shares = rqdatac.get_shares(stock_list,start_date=date, end_date=date, fields='total')

    share_price = rqdatac.get_price(stock_list,start_date=date,end_date=date,fields='close')

    operating_cash_per_share = cash_flow_from_operating_activities_ttm / total_shares

    cash_earning_to_price= operating_cash_per_share.T/share_price.T

    processed_cash_earning =  winsorization_and_market_cap_weighed_standardization(cash_earning_to_price[date], market_cap_on_current_day)

    return processed_cash_earning


# book-to-price = (股东权益合计-优先股)/市值
def book_to_price_ratio(date,market_cap_on_current_day):

    total_equity = lf(rqdatac.financials.balance_sheet.total_equity,date)
    # 大多公司没有优先股，优先股空值比例高达98%，进行缺失值处理将空值替换为0
    prefer_stock = lf(rqdatac.financials.balance_sheet.equity_prefer_stock,date)
    prefer_stock = prefer_stock.fillna(value=0)

    bp_ratio = (total_equity-prefer_stock)/market_cap_on_current_day[total_equity.index]
    btop = winsorization_and_market_cap_weighed_standardization(bp_ratio, market_cap_on_current_day)

    return btop


# style:leverage
# MLEV: Market leverage = (ME+PE+LD)/ME ME:最新市值 PE:最新优先股账面价值 LD:长期负债账面价值
# 根据Barra 因子解释：total debt=long term debt+current liabilities,在因子计算中使用非流动负债合计：non_current_liabilities作为long_term_debt
def get_market_leverage(date,market_cap_on_current_day):

    non_current_liabilities = lf(rqdatac.financials.balance_sheet.non_current_liabilities,date)
    non_current_liabilities = non_current_liabilities.fillna(value=0)
    # 大多公司没有优先股，优先股空值比例高达98%，进行缺失值处理将空值替换为0
    prefer_stock = lf(rqdatac.financials.balance_sheet.equity_prefer_stock, date)
    prefer_stock = prefer_stock.fillna(value=0)

    MLEV = (market_cap_on_current_day[non_current_liabilities.index]+non_current_liabilities+prefer_stock)/market_cap_on_current_day[non_current_liabilities.index]
    processed_MLEV = winsorization_and_market_cap_weighed_standardization(MLEV, market_cap_on_current_day)

    return processed_MLEV


# DTOA:Debt_to_asset：total debt/total assets
def get_debt_to_asset(date,market_cap_on_current_day):

    total_debt = lf(rqdatac.financials.balance_sheet.total_liabilities,date)

    total_asset = lf(rqdatac.financials.balance_sheet.total_assets,date)
    dtoa = total_debt/total_asset
    processed_dtoa = winsorization_and_market_cap_weighed_standardization(dtoa, market_cap_on_current_day)

    return processed_dtoa


# BLEV: book leverage = (BE+PE+LD)/BE BE:普通股权账面价值 PE：优先股账面价值 LD:长期负债账面价值
# 由于BE=total equity-equity_prefer_stock
def get_book_leverage(date,market_cap_on_current_day):

    book_value_of_common_stock = lf(rqdatac.financials.balance_sheet.paid_in_capital,date)

    non_current_liabilities = lf(rqdatac.financials.balance_sheet.non_current_liabilities, date)
    non_current_liabilities = non_current_liabilities.fillna(value=0)

    # 大多公司没有优先股，优先股空值比例高达98%，进行缺失值处理将空值替换为0
    prefer_stock = lf(rqdatac.financials.balance_sheet.equity_prefer_stock, date)
    prefer_stock = prefer_stock.fillna(value=0)

    blev = (book_value_of_common_stock+prefer_stock+non_current_liabilities)/book_value_of_common_stock
    processed_blev = winsorization_and_market_cap_weighed_standardization(blev, market_cap_on_current_day)

    return processed_blev


def get_sales_growth(date,year,market_cap_on_current_day):
    recent_report, annual_report, annual_report_last_year, annual_report_2_year_ago, annual_report_3_year_ago, annual_report_4_year_ago = last_five_annual_report(
        date)
    growth_listed_date_threshold = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1825)).strftime("%Y-%m-%d")
    growth_qualified_stocks = [i for i in annual_report.index.tolist() if
                               rqdatac.instruments(i).listed_date < growth_listed_date_threshold]

    factor = pd.DataFrame(index=growth_qualified_stocks, columns=['SGRO'])

    for stock in growth_qualified_stocks:
        query = rqdatac.query(rqdatac.financials.income_statement.operating_revenue).filter(
            rqdatac.financials.stockcode.in_([stock]))
        sales_recent = rqdatac.get_financials(query, annual_report[stock], '1q')

        latest_trading_date_recent = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_recent = rqdatac.get_shares(stock, start_date=latest_trading_date_recent,
                                           end_date=latest_trading_date_recent, fields='total')

        sales_per_share_recent = sales_recent.values / shares_recent.values

        sales_last_year = rqdatac.get_financials(query, annual_report_last_year[stock], '1q')

        latest_trading_date_last_year = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_last_year[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_last_year = rqdatac.get_shares(stock, start_date=latest_trading_date_last_year,
                                              end_date=latest_trading_date_last_year, fields='total')

        sales_per_share_last_year = sales_last_year.values / shares_last_year.values

        sales_2_year_ago = rqdatac.get_financials(query, annual_report_2_year_ago[stock], '1q')

        latest_trading_date_2_year_ago = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_2_year_ago[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_2_year_ago = rqdatac.get_shares(stock, start_date=latest_trading_date_2_year_ago,
                                               end_date=latest_trading_date_2_year_ago, fields='total')

        sales_per_share_2_year_ago = sales_2_year_ago.values / shares_2_year_ago.values

        sales_3_year_ago = rqdatac.get_financials(query, annual_report_3_year_ago[stock], '1q')

        latest_trading_date_3_year_ago = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_3_year_ago[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_3_year_ago = rqdatac.get_shares(stock, start_date=latest_trading_date_3_year_ago,
                                               end_date=latest_trading_date_3_year_ago, fields='total')

        sales_per_share_3_year_ago = sales_3_year_ago.values / shares_3_year_ago.values

        sales_4_year_ago = rqdatac.get_financials(query, annual_report_4_year_ago[stock], '1q')

        latest_trading_date_4_year_ago = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_4_year_ago[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_4_year_ago = rqdatac.get_shares(stock, start_date=latest_trading_date_4_year_ago,
                                               end_date=latest_trading_date_4_year_ago, fields='total')

        sales_per_share_4_year_ago = sales_4_year_ago.values / shares_4_year_ago.values

        regression = linear_model.LinearRegression()
        sales_per_share = np.array(
            [sales_per_share_recent, sales_per_share_last_year, sales_per_share_2_year_ago, sales_per_share_3_year_ago,
             sales_per_share_4_year_ago])
        regression.fit(year.reshape(-1, 1), sales_per_share)
        factor['SGRO'][stock] = float(regression.coef_) / abs(sales_per_share).mean()

    sale_growth = winsorization_and_market_cap_weighed_standardization(factor['SGRO'], market_cap_on_current_day)

    return sale_growth


def get_earnings_growth(date,year,market_cap_on_current_day):
    recent_report, annual_report, annual_report_last_year, annual_report_2_year_ago, annual_report_3_year_ago, annual_report_4_year_ago = last_five_annual_report(
        date)
    growth_listed_date_threshold = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1825)).strftime("%Y-%m-%d")
    growth_qualified_stocks = [i for i in annual_report.index.tolist() if
                               rqdatac.instruments(i).listed_date < growth_listed_date_threshold]

    factor = pd.DataFrame(index=growth_qualified_stocks, columns=['EGRO'])

    for stock in growth_qualified_stocks:
        # 实际操作中发现有部分公司会在财报发布后对报表进行多次调整，调整后eps为空，比如'601519.XSHG'，该公司报表在发布后经过多次调整，2014年年报主要财务指标表"基本eps"数据缺失，但是在利润表中"基本eps"数据存在，
        # 所以在取数据时进行判断，如果financial_indicator为首选表，income_statement 为备选表
        query_f = rqdatac.query(rqdatac.financials.financial_indicator.earnings_per_share).filter(
            rqdatac.financials.stockcode.in_([stock]))

        query_i = rqdatac.query(rqdatac.financials.income_statement.basic_earnings_per_share).filter(
            rqdatac.financials.stockcode.in_([stock]))

        eps_recent = rqdatac.get_financials(query_f, annual_report[stock], '1q') if \
            rqdatac.get_financials(query_f, annual_report[stock], '1q').isnull().sum() == 0 \
            else rqdatac.get_financials(query_i, annual_report[stock], '1q')
        eps_last_year = rqdatac.get_financials(query_f, annual_report_last_year[stock], '1q') if \
            rqdatac.get_financials(query_f, annual_report_last_year[stock], '1q').isnull().sum() == 0 \
            else rqdatac.get_financials(query_i, annual_report_last_year[stock], '1q')
        eps_2_year_ago = rqdatac.get_financials(query_f, annual_report_2_year_ago[stock], '1q') if \
            rqdatac.get_financials(query_f, annual_report_2_year_ago[stock], '1q').isnull().sum() == 0 \
            else rqdatac.get_financials(query_i, annual_report_2_year_ago[stock], '1q')
        eps_3_year_ago = rqdatac.get_financials(query_f, annual_report_3_year_ago[stock], '1q') if \
            rqdatac.get_financials(query_f, annual_report_3_year_ago[stock], '1q').isnull().sum() == 0 \
            else rqdatac.get_financials(query_i, annual_report_3_year_ago[stock], '1q')
        eps_4_year_ago = rqdatac.get_financials(query_f, annual_report_4_year_ago[stock], '1q') if \
            rqdatac.get_financials(query_f, annual_report_4_year_ago[stock], '1q').isnull().sum() == 0 \
            else rqdatac.get_financials(query_i, annual_report_4_year_ago[stock], '1q')

        regression = linear_model.LinearRegression()
        eps = np.array(
            [eps_recent, eps_last_year, eps_2_year_ago, eps_3_year_ago, eps_4_year_ago])
        regression.fit(year.reshape(-1, 1), eps)
        factor['EGRO'][stock] = float(regression.coef_) / abs(eps.mean())
    earning_growth = winsorization_and_market_cap_weighed_standardization(factor['EGRO'], market_cap_on_current_day)

    return earning_growth


date = '2018-02-12'
year = pd.Series([48, 36, 24, 12, 0])


def get_style_factors(date):

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_252_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-252]

    stock_list = rqdatac.all_instruments(type = 'CS', date = latest_trading_date)['order_book_id'].values.tolist()

    stock_excess_return, market_portfolio_excess_return = get_daily_excess_return(stock_list, trading_date_252_before.strftime('%Y-%m-%d'), latest_trading_date.strftime('%Y-%m-%d'))

    market_cap_on_current_day = rqdatac.get_factor(id_or_symbols = stock_excess_return.columns.tolist(), factor = 'market_cap', start_date = latest_trading_date.strftime('%Y-%m-%d'), end_date = latest_trading_date.strftime('%Y-%m-%d'))

    size_exposure = size(market_cap_on_current_day)

    non_linear_size_exposure = non_linear_size(size_exposure, market_cap_on_current_day)

    market_portfolio_beta, market_portfolio_beta_exposure = get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day)

    daily_standard_deviation_exposure = get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day)

    cumulative_range_exposure = get_cumulative_range(stock_list, latest_trading_date, market_cap_on_current_day)

    historical_sigma_exposure = get_historical_sigma(stock_excess_return, market_portfolio_excess_return, market_portfolio_beta, market_portfolio_beta_exposure, market_cap_on_current_day)

    residual_volatility_exposure = 0.74*daily_standard_deviation_exposure + 0.16*cumulative_range_exposure + 0.1*historical_sigma_exposure

    momentum_exposure = get_momentum(stock_list, latest_trading_date, market_cap_on_current_day)

    liquidity_exposure = get_liquidity(stock_list, latest_trading_date, market_cap_on_current_day)

    earnings_to_price = get_earning_to_price_ratio(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    cash_earnings_to_price = get_cash_earnings_to_price_ratio(latest_trading_date.strftime('%Y-%m-%d'),market_cap_on_current_day)

    earnings_yield = earnings_to_price*(11/32)+cash_earnings_to_price*(21/32)
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


