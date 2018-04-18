import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import rqdatac

#rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))
rqdatac.init('ricequant', '8ricequant8',('q-tools.ricequant.com', 16010))


def recent_annual_report(date):
    latest_trading_date = str(
        rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))

    previous_year = datetime.strptime(latest_trading_date, '%Y-%m-%d').year - 1

    # 取出最近一期财务报告类型，例如 '2016q3' 或  '2016q4'， 其中 '2016q3' 表示前三季度累计； '2016q4' 表示年报

    recent_report_type = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.income_statement.net_profit),
                                                  entry_date=latest_trading_date, interval='1y', report_quarter=True)['report_quarter']

    annual_report_type = recent_report_type.copy()  # 深拷贝

    # 若上市公司未发布今年的财报，且未发布去年的年报，则取前年的年报为最新年报

    if recent_report_type.T.iloc[0].values[0][:4] == str(previous_year):

        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year - 1) + 'q4'

    # 若上市公司已发布今年的财报，则取去年的年报为最新年报

    else:
        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year) + 'q4'

    # recent_report_type 和 annual_report_type 均为 dataframe 格式，输出时转为 Series 格式

    return recent_report_type.T[latest_trading_date], annual_report_type.T[latest_trading_date]


# 计算原生指标过去十二个月的滚动值（利润表、现金流量表滚动求和）
def ttm_sum(financial_indicator, date):

    recent_report_type, annual_report_type = recent_annual_report(date)
    latest_trading_date = str(rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))
    previous_year = datetime.strptime(latest_trading_date, '%Y-%m-%d').year - 1

    # 获得最近一期报告为年报的股票列表

    annual_report_published_stocks = recent_report_type[recent_report_type == str(previous_year) + 'q4'].index.tolist()

    # 把 index 和 list 转为集合类型，再计算补集

    annual_report_not_published_stocks = list(set(recent_report_type.index) - set(annual_report_published_stocks))

    # TTM 计算对于未发布年报的企业仅考虑上市时间超过半年（183天）的股票(考虑到招股说明书披露相关财报)，以保证获得相关财务数据进行计算
    ttm_listed_date_threshold = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=183)).strftime("%Y-%m-%d")

    ttm_qualified_stocks = [i for i in annual_report_not_published_stocks if
                            rqdatac.instruments(i).listed_date < ttm_listed_date_threshold]

    # 计算最近一期财报为年报的股票的TTM
    annual_published_recent_annual_values = [
        rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_([stock])),
                               annual_report_type[stock], '1q').values[0] for stock in annual_report_published_stocks]

    annual_published_ttm_series = pd.Series(index=annual_report_published_stocks,
                                            data=annual_published_recent_annual_values)

    # 对于最近一期报告非年报的股票，获取其此前同期的报告
    previous_same_period = str(int(recent_report_type[0][:4]) - 1)

    previous_same_period_report_type = recent_report_type.loc[ttm_qualified_stocks].str.slice_replace(0, 4,previous_same_period)

    # 计算最近一期财报不是年报的股票的TTM

    # 最近一期季报/半年报取值

    recent_values = [
        rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_([stock])),
                               recent_report_type[stock], '1q').values[0] for stock in ttm_qualified_stocks]

    # 去年同期季报/半年报取值

    previous_same_period_values = [
        rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_([stock])),
                               previous_same_period_report_type[stock], '1q').values[0] for stock in ttm_qualified_stocks]

    # 最近一期年报报告取值

    recent_annual_values = [
        rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_([stock])),
                               annual_report_type[stock], '1q').values[0] for stock in ttm_qualified_stocks]

    ttm_values = np.array(recent_annual_values) + np.array(recent_values) - np.array(previous_same_period_values)

    annual_not_published_ttm_series = pd.Series(index=ttm_qualified_stocks, data=ttm_values)

    ttm_series = pd.concat([annual_published_ttm_series, annual_not_published_ttm_series], axis=0)

    return ttm_series


# 调取最近一期财报数据
def lf(financial_indicator, date):

    recent_report_type, annual_report_type = recent_annual_report(date)

    recent_annual_values = [rqdatac.get_financials(rqdatac.query(financial_indicator).filter(rqdatac.financials.stockcode.in_([stock])),
                                                   recent_report_type[stock], '1q').values[0] for stock in recent_report_type.index]

    lf_series = pd.Series(index=recent_report_type.index, data=recent_annual_values)

    return lf_series


# style: earnings yield
# ETOP:Trailing earning-to-price ratio
def earning_to_price_ratio(date):

    # 获取最近一个交易日
    latest_trading_date = str(rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))

    # 行情数据部分取最近一个交易日数据

    market_cap_series = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.eod_derivative_indicator.market_cap),
                                                 entry_date=latest_trading_date, interval='1d')['market_cap']

    # 财务数据跟据情况按 TTM, LYR 和 LF 三种方法计算，EP只需要计算TTM和LYR两种方法

    net_profit_ttm = ttm_sum(rqdatac.financials.income_statement.net_profit,date)

    ep_ratio = net_profit_ttm/market_cap_series[net_profit_ttm.index]

    return ep_ratio.T


# CETOP:Trailing cash earning to price ratio
def cash_earnings_to_price_ratio(date):

    # 获取最近一个交易日
    latest_trading_date = str(
        rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))

    cash_flow_from_operating_activities_ttm = ttm_sum(rqdatac.financials.cash_flow_statement.cash_flow_from_operating_activities,date)

    total_shares_ttm = rqdatac.get_shares(cash_flow_from_operating_activities_ttm.index.tolist(), start_date=latest_trading_date, end_date=latest_trading_date, fields='total')

    operating_cash_flow_per_share = cash_flow_from_operating_activities_ttm / total_shares_ttm

    return operating_cash_flow_per_share.T


# style: book-to-price  BTOP =股东权益合计/市值
def book_to_price_ratio_total(date):
    # 获取最近一个交易日
    latest_trading_date = str(
        rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))

    # 行情数据部分取最近一个交易日数据

    market_cap_series = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.eod_derivative_indicator.market_cap),
                                                 entry_date=latest_trading_date, interval='1d')['market_cap']

    total_equity = lf(rqdatac.financials.balance_sheet.total_equity,date)

    bp_ratio = total_equity/market_cap_series[total_equity.index]

    return bp_ratio.T


# book-to-price = (股东权益合计-优先股)/市值
def book_to_price_ratio(date):
    # 获取最近一个交易日
    latest_trading_date = str(
        rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))

    # 行情数据部分取最近一个交易日数据

    market_cap_series = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.eod_derivative_indicator.market_cap),
                                                 entry_date=latest_trading_date, interval='1d')['market_cap']

    total_equity = lf(rqdatac.financials.balance_sheet.total_equity,date)
    # 大多公司没有优先股，优先股空值比例高达98%，进行缺失值处理将空值替换为0
    prefer_stock = lf(rqdatac.financials.balance_sheet.equity_prefer_stock,date)
    prefer_stock = prefer_stock.fillna(value=0)

    bp_ratio = (total_equity-prefer_stock)/market_cap_series[total_equity.index]

    return bp_ratio.T


# style:leverage
# MLEV: Market leverage
def market_leverage(date):
    # 获取最近一个交易日
    latest_trading_date = str(
        rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))

    # 行情数据部分取最近一个交易日数据

    market_cap_series = \
    rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.eod_derivative_indicator.market_cap),
                             entry_date=latest_trading_date, interval='1d')['market_cap']

    long_term_debt = lf(rqdatac.financials.balance_sheet.long_term_liabilities,date)
    # 大多公司没有优先股，优先股空值比例高达98%，进行缺失值处理将空值替换为0
    prefer_stock = lf(rqdatac.financials.balance_sheet.equity_prefer_stock, date)
    prefer_stock = prefer_stock.fillna(value=0)

    MLEV = (market_cap_series[long_term_debt.index]+long_term_debt+prefer_stock)/market_cap_series[long_term_debt.index]
    return MLEV.T


# DTOA:Debt_to_asset
def debt_to_asset(date):

    total_debt = lf(rqdatac.financials.balance_sheet.total_liabilities,date)

    total_asset = lf(rqdatac.financials.balance_sheet.total_assets,date)

    return pd.DataFrame(total_debt/total_asset)


# BLEV: book leverage
def book_leverage(date):

    total_equity = lf(rqdatac.financials.balance_sheet.total_equity,date)

    long_term_debt = lf(rqdatac.financials.balance_sheet.long_term_liabilities, date)

    # 优先股book value缺失
    # 大多公司没有优先股，优先股空值比例高达98%，进行缺失值处理将空值替换为0
    prefer_stock = lf(rqdatac.financials.balance_sheet.equity_prefer_stock, date)
    prefer_stock = prefer_stock.fillna(value=0)

    BLEV = (total_equity+long_term_debt+prefer_stock)/total_equity

    return pd.DataFrame(BLEV)


date = '2018-02-02'
etop = earning_to_price_ratio(date)
cetop = cash_earnings_to_price_ratio(date)
btop = book_to_price_ratio(date)
btop_total = book_to_price_ratio_total(date)
mlev = market_leverage(date)
dtoa = debt_to_asset(date)
blev = book_leverage(date)

leverage = pd.DataFrame(index=mlev.index,columns=['leverage'])

btop.to_csv('/Users/rice/Desktop/Barra factor/btop.csv', index=True, na_rep='NaN', header=True)
btop_total.to_csv('/Users/rice/Desktop/Barra factor/btop_total.csv', index=True, na_rep='NaN', header=True)

earnings_yield = pd.DataFrame(index=etop.index,columns=['earnings_yield'])
earnings_yield['earnings_yield']=[etop.loc[stock].values[0]*(11/32)+cetop.loc[stock].values[0]*(21/32) for stock in earnings_yield.index.tolist()]
earnings_yield.to_csv('/Users/rice/Desktop/Barra factor/earnings_yield.csv', index=True, na_rep='NaN', header=True)

# leverage缺失值过多，需要对缺失值进行处理
# 根据Barra的缺失值处理逻辑对leverage缺失值进行处理
for stock in mlev.index.tolist():
    if str(mlev.loc[stock].values[0]) != 'nan':
        if str(dtoa.loc[stock].values[0]) != 'nan':
            if str(blev.loc[stock].values[0]) != 'nan':
                leverage['leverage'][stock] = mlev.loc[stock].values[0]*0.38+dtoa.loc[stock].values[0]*0.35+blev.loc[stock].values[0]*0.27
            else:
                leverage['leverage'][stock] = mlev.loc[stock].values[0]*(38/73)+dtoa.loc[stock].values[0]*(35/73)
        elif str(blev.loc[stock].values[0]) != 'nan':
            leverage['leverage'][stock] = mlev.loc[stock].values[0] * (38 / 65) + blev.loc[stock].values[0] * (27 / 65)
        else:
            leverage['leverage'][stock] = mlev.loc[stock].values[0]
    else:
        if str(dtoa.loc[stock].values[0]) != 'nan':
            if str(blev.loc[stock].values[0]) != 'nan':
                leverage['leverage'][stock] = dtoa.loc[stock].values[0] * (35 / 62) + blev.loc[stock].values[0] * (27 / 62)
            else:
                leverage['leverage'][stock] = dtoa.loc[stock].values[0]
        else:
            leverage['leverage'][stock] = blev.loc[stock].values[0]

leverage.to_csv('/Users/rice/Desktop/Barra factor/leverage.csv', index=True, na_rep='NaN', header=True)








