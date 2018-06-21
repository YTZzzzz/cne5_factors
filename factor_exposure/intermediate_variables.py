import sys

import numpy as np
import pandas as pd
from datetime import datetime

import rqdatac
rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))

sys.path.append("/Users/rice/Documents/cne5_factors/factor_exposure/")


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))[::-1]


def risk_free_rate(date):

    shibor_now = pd.Series(ts.shibor_data(date.year)['3M'].values, index=ts.shibor_data(date.year)['date'])

    shibor_last_year = pd.Series(ts.shibor_data(date.year-1)['3M'].values, index=ts.shibor_data(date.year-1)['date'])

    shibor_2years_ago = pd.Series(ts.shibor_data(date.year-2)['3M'].values, index=ts.shibor_data(date.year-2)['date'])

    RF_rate = pd.concat([shibor_2years_ago,shibor_last_year,shibor_now], axis=0)

    return RF_rate


def get_market_portfolio_return(filtered_stock_daily_return, market_cap_on_current_day):

    market_cap_filtered_universe = market_cap_on_current_day[market_cap_on_current_day > 3000000000].index.tolist()

    # 取收益率缺失情况，及市值均符合要求的股票池交集

    qualified_universe = list(set(filtered_stock_daily_return.columns.tolist()) & set(market_cap_filtered_universe))

    market_portfolio_daily_return = pd.Series(np.diag(filtered_stock_daily_return[qualified_universe].replace(np.nan, 0).dot((market_cap_on_current_day[qualified_universe].replace(np.nan, 0)/market_cap_on_current_day.sum()).T)), index=qualified_universe)

    return market_portfolio_daily_return


def get_daily_excess_return(stock_list, market_cap_on_current_day, start_date, end_date):

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    stock_daily_return = rqdatac.get_price(stock_list, rqdatac.get_previous_trading_date(start_date), end_date, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据存在空值的股票

    inds = stock_daily_return.isnull().sum()[stock_daily_return.isnull().sum() > 0].index

    filtered_stock_daily_return = stock_daily_return.drop(inds, axis=1)

    # 目前的实现存在问题，应该每一天提取当天上市的股票，以及上市公司的市值，筛选 market portfolio 的股票池和进行收益率计算。待 Barra 确认相关计算方法后，再进行预计算。

    #market_portfolio_daily_return = get_market_portfolio_return(filtered_stock_daily_return, market_cap_on_current_day)

    # 经测试发现，中证全指（000985）作为 market portfolio 的效果最好

    market_portfolio_daily_return = rqdatac.get_price('000985.XSHG', rqdatac.get_previous_trading_date(start_date), end_date, frequency='1d', fields='close').pct_change()[1:]

    # 计算无风险日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=start_date, end_date=end_date, tenor='3M')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[filtered_stock_daily_return.index]

    daily_excess_return = filtered_stock_daily_return.T.subtract(risk_free_return.iloc[:,0]).T

    market_portfolio_daily_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:,0])

    return daily_excess_return, market_portfolio_daily_excess_return


def get_recent_financial_report(date):

    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

    # 取出最近一期财务报告类型，例如 '2016q3' 或  '2016q4'， 其中 '2016q3' 表示前三季度累计； '2016q4' 表示年报

    recent_report_type = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.income_statement.net_profit), entry_date=date, report_quarter=True)['report_quarter']

    annual_report_type = recent_report_type.copy()  # 深拷贝

    # 若上市公司未发布去年的财报，则取前年的年报为最新年报

    if recent_report_type.T.iloc[0].values[0][:4] == str(previous_year):

        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year - 1) + 'q4'

    else:
        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year) + 'q4'

    # recent_report_type 和 annual_report_type 均为 dataframe 格式，输出时转为 Series 格式

    return recent_report_type.T[date], annual_report_type.T[date]


def get_recent_five_annual_shares(stock_list, date):

    # 上市公司每年4月30日前必须公布当年报告。因此，取此前每年5月1日后第一个交易日的股票A股流通股本，作为当年的股本

    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

    month = datetime.strptime(date, '%Y-%m-%d').month

    list_of_dates = [str(previous_year) + '-05-01', str(previous_year - 1) + '-05-01',str(previous_year - 2) + '-05-01', str(previous_year - 3) + '-05-01',str(previous_year - 4) + '-05-01'] \
        if month > 5 \
        else [str(previous_year-1) + '-05-01', str(previous_year - 2) + '-05-01',str(previous_year - 3) + '-05-01', str(previous_year - 4) + '-05-01',str(previous_year - 5) + '-05-01']

    recent_five_annual_shares = pd.DataFrame()

    for report_date in list_of_dates:

        next_trading_date = rqdatac.get_next_trading_date(report_date)

        recent_five_annual_shares[report_date] = rqdatac.get_shares(stock_list, start_date = next_trading_date.strftime('%Y-%m-%d'), end_date = next_trading_date.strftime('%Y-%m-%d'), fields='total_a').iloc[0]

    # 调整股本 dataframe 的列名，方便相除计算每股收入

    recent_five_annual_shares.columns = ['first', 'second', 'third', 'fourth', 'fifth']

    return recent_five_annual_shares


# 计算原生指标过去十二个月的滚动值（利润表、现金流量表滚动求和）

def get_ttm_sum(financial_indicator, date, recent_report_type, annual_report_type):

    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

    month_now = datetime.strptime(date, '%Y-%m-%d').month

    # 获得最近一期报告为年报的股票列表

    annual_report_published_stocks = recent_report_type[recent_report_type == str(previous_year) + 'q4'].index.tolist()

    # 把 index 和 list 转为集合类型，再计算补集

    annual_report_not_published_stocks = list(set(recent_report_type.index) - set(annual_report_published_stocks))

    # 计算最近一期财报为年报的股票的TTM

    annual_published_recent_annual_values = [rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_([stock])), annual_report_type[stock], '1q').values[0] for stock in annual_report_published_stocks]

    annual_published_ttm_values = pd.Series(index=annual_report_published_stocks, data=annual_published_recent_annual_values)

    # 计算最近一期财报不是年报的股票的TTM

    # 获取最近五期财报的财务数据

    recent_five_reports = rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_(annual_report_not_published_stocks)), recent_report_type[0], '5q')

    # 对于最近一期报告不是年报的上市公司，其财务数据的 TTM 值为（最近一期年报财务数据 + 最近一期报告财务数据 - 去年同期报告财务数据）

    recent_values = recent_five_reports.iloc[0]

    recent_annual_values = [recent_five_reports.loc[str(previous_year) + 'q4'] if month_now >= 5 else recent_five_reports.loc[str(previous_year - 1) + 'q4']][0]

    previous_same_period_values = recent_five_reports.iloc[-1]

    annual_not_published_ttm_values = recent_annual_values + recent_values - previous_same_period_values

    ttm_series = pd.concat([annual_published_ttm_values, annual_not_published_ttm_values], axis=0)

    return ttm_series


# 调取最近一期财报数据

def get_last_reported_values(financial_indicator, recent_report_type):

    # 取出当天所有出现的财报类型

    unique_recent_report_type = recent_report_type.unique().tolist()

    last_reported_values = pd.Series()

    # 循环每一类型的报告，再合并返回

    for report_type in unique_recent_report_type:

        stock_list = recent_report_type[recent_report_type == report_type].index.tolist()

        if len(stock_list) == 1:

            last_reported_values = last_reported_values.append(rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_(stock_list)), report_type))

        else:

            last_reported_values = last_reported_values.append(rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_(stock_list)), report_type).iloc[0])

    return last_reported_values


def recent_five_annual_values(financial_indicator, date, recent_report_type):

    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

    # 获得最近一期报告为年报的股票列表

    annual_report_published_stocks = recent_report_type[recent_report_type == str(previous_year) + 'q4'].index.tolist()

    # 把 index 和 list 转为集合类型，再计算补集

    annual_report_not_published_stocks = list(set(recent_report_type.index) - set(annual_report_published_stocks))

    # 对于去年年报已经发布的上市公司，最近五期年报的列表

    annual_report_published_list = [str(previous_year) + 'q4', str(previous_year - 1) + 'q4', str(previous_year - 2) + 'q4', str(previous_year - 3) + 'q4', str(previous_year - 4) + 'q4']

    # 对于去年年报尚未经发布的上市公司，最近五期年报的列表

    annual_report_not_published_list = [str(previous_year - 1) + 'q4', str(previous_year - 2) + 'q4', str(previous_year - 3) + 'q4', str(previous_year - 4) + 'q4', str(previous_year - 5) + 'q4']

    # 获得最近一期报告为年报的股票列表

    recent_five_reports = rqdatac.get_financials(rqdatac.query(financial_indicator), str(previous_year) + 'q4', '25q').T

    annual_report_published_values = recent_five_reports[annual_report_published_list].loc[annual_report_published_stocks]

    annual_report_not_published_values = recent_five_reports[annual_report_not_published_list].loc[annual_report_not_published_stocks]

    # 重新命名 columns，方便合并 dataframes

    annual_report_published_values.columns = ['first', 'second', 'third', 'fourth', 'fifth']

    annual_report_not_published_values.columns = ['first', 'second', 'third', 'fourth', 'fifth']

    recent_five_reports_values = pd.concat([annual_report_published_values, annual_report_not_published_values], axis = 0)

    return recent_five_reports_values


def get_financial_and_market_data(stock_list, latest_trading_date, trading_date_252_before):

    # 取出最近一期财务报告和年度报告字段，例如 '2016q3' 或  '2016q4'

    recent_report_type, annual_report_type = get_recent_financial_report(latest_trading_date.strftime('%Y-%m-%d'))

    market_cap_on_current_day = rqdatac.get_factor(id_or_symbols=stock_list, factor='a_share_market_val', start_date=latest_trading_date.strftime('%Y-%m-%d'), end_date=latest_trading_date.strftime('%Y-%m-%d'))

    stock_excess_return, market_portfolio_excess_return = get_daily_excess_return(stock_list, market_cap_on_current_day, trading_date_252_before.strftime('%Y-%m-%d'), latest_trading_date.strftime('%Y-%m-%d'))

    recent_five_annual_shares = get_recent_five_annual_shares(stock_list, latest_trading_date.strftime('%Y-%m-%d'))

    # 当公司非流动性负债数据缺失时，则认为该公司没有非流动性负债，把缺失值替换为0

    last_reported_non_current_liabilities = get_last_reported_values(rqdatac.financials.balance_sheet.non_current_liabilities, recent_report_type).fillna(value=0)

    # 当公司优先股数据缺失时，则认为该公司没有优先股，把缺失值替换为0

    last_reported_preferred_stock = get_last_reported_values(rqdatac.financials.balance_sheet.equity_prefer_stock, recent_report_type).fillna(value=0)

    return recent_report_type, annual_report_type, market_cap_on_current_day, stock_excess_return, market_portfolio_excess_return, recent_five_annual_shares, last_reported_non_current_liabilities, last_reported_preferred_stock


def get_shenwan_industry_label(stock_list, date):

    industry_classification = rqdatac.shenwan_instrument_industry(stock_list, date)

    industry_classification_missing_stocks = list(set(stock_list) - set(industry_classification.index.tolist()))

    # 当传入股票过多时，对于缺失行业标记的股票，RQD 目前不会向前搜索，因此需要循环单个传入查找这些股票的行业标记

    if len(industry_classification_missing_stocks) != 0:

        for stock in industry_classification_missing_stocks:

            missing_industry_classification = rqdatac.shenwan_instrument_industry(stock, date)

            if missing_industry_classification != None:

                industry_classification = industry_classification.append(pd.Series([missing_industry_classification[0], missing_industry_classification[1]], index=['index_code','index_name'], name = stock))

    return industry_classification['index_name']
