
import rqdatac

rqdatac.init('ricequant', '8ricequant8',('q-tools.ricequant.com', 16010))


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))[::-1]


# 剔除垃圾股
def drop_st_stock(stock_list,date):

    st_stock_list = rqdatac.is_st_stock(stock_list,start_date=date,end_date=date)

    stock_list_drop_st = [stock for stock in stock_list if st_stock_list.T.loc[stock].values[0] == False]

    return stock_list_drop_st


# 剔除停牌股
def drop_suspended_stock(stock_list,date):

    stock_list_drop_suspended = [stock for stock in stock_list if rqdatac.is_suspended(stock,start_date=date,end_date=date)[stock].values[0] == False]

    return stock_list_drop_suspended


def drop_stock(days,date,stock_list):

    threshold_date = rqdatac.get_trading_dates(date - timedelta(days=days*2), date, country='cn')[-days]

    stock_list = [stock for stock in stock_list if rqdatac.instruments(stock).listed_date < str(threshold_date)]


    return stock_list


def risk_free_rate(date):

    shibor_now = pd.Series(ts.shibor_data(date.year)['3M'].values, index=ts.shibor_data(date.year)['date'])

    shibor_last_year = pd.Series(ts.shibor_data(date.year-1)['3M'].values, index=ts.shibor_data(date.year-1)['date'])

    shibor_2years_ago = pd.Series(ts.shibor_data(date.year-2)['3M'].values, index=ts.shibor_data(date.year-2)['date'])

    RF_rate = pd.concat([shibor_2years_ago,shibor_last_year,shibor_now], axis=0)

    return RF_rate


'''
def get_daily_excess_return(stock_list, start_date, end_date):

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）
    end_date = str(end_date)

    daily_return = rqdatac.get_price(stock_list, rqdatac.get_previous_trading_date(start_date), end_date, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    sw_name = rqdatac.shenwan_instrument_industry(stock_list)['index_name']

    market_list = [stock for stock in stock_list if rqdatac.instruments(stock).listed_date < end_date]

    market_cap = rqdatac.get_factor(id_or_symbols = market_list, factor = 'a_share_market_val', start_date = start_date, end_date = end_date)

    shenwan_with_market_cap = pd.concat([sw_name,market_cap.T[end_date]],axis=1)
    shenwan_with_market_cap.columns = ['index_name','market_cap']

    industry_list = []
    for group in shenwan_with_market_cap.groupby('index_name'):
        group_list = group[1].index.tolist()
        industry_list.extend(list(rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.eod_derivative_indicator.market_cap).order_by(rqdatac.fundamentals.eod_derivative_indicator.market_cap.desc()).limit(len(group_list)*0.3),entry_date=end_date).minor_axis))

    market_cap_industry = rqdatac.get_factor(id_or_symbols = industry_list, factor = 'a_share_market_val', start_date = start_date, end_date = end_date)

    market_daily_return = rqdatac.get_price(industry_list, rqdatac.get_previous_trading_date(start_date), end_date, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    market_portfolio_daily_return = pd.Series(np.diag(market_daily_return.replace(np.nan, 0).dot((market_cap_industry.replace(np.nan, 0)/market_cap_industry.sum()).T)), index=daily_return.index)

    # 计算无风险日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=start_date, end_date=end_date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1).loc[daily_return.index]

    daily_excess_return = daily_return.T.subtract(risk_free_return.iloc[:,0]).T

    market_portfolio_daily_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:,0])

    # 剔除收益率数据少于66个的股票
    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > (len(daily_return) - 66)].index

    return daily_excess_return.drop(daily_excess_return[inds], axis=1), market_portfolio_daily_excess_return
'''


# 使用中证全指作为market portfolio
def get_daily_excess_return(stock_list, start_date, end_date):

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）
    stock_list = [stock for stock in stock_list if rqdatac.instruments(stock).listed_date < end_date]

    daily_return = rqdatac.get_price_change_rate(stock_list,start_date,end_date)

    market_portfolio_daily_return = rqdatac.get_price_change_rate('000985.XSHG',start_date,end_date)

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=start_date, end_date=end_date, tenor='0S')
    risk_free_return = (((1 + compounded_risk_free_return) ** (1/365)) - 1).loc[daily_return.index]

    daily_excess_return = daily_return.T.subtract(risk_free_return.iloc[:, 0]).T

    market_portfolio_daily_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:, 0])

    # 剔除收益率数据少于66个的股票
    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > (len(daily_return) - 66)].index

    return daily_excess_return.drop(daily_excess_return[inds], axis=1), market_portfolio_daily_excess_return


def get_daily_excess_return(stock_list, start_date, end_date):

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    daily_return = rqdatac.get_price(stock_list, rqdatac.get_previous_trading_date(start_date), end_date, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    market_list = list(rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.eod_derivative_indicator.a_share_market_val_2).filter(rqdatac.fundamentals.eod_derivative_indicator.market_cap > 10000000000),entry_date=end_date).minor_axis)

    market_list = [stock for stock in market_list if rqdatac.instruments(stock).listed_date < end_date]

    market_daily_return = rqdatac.get_price(market_list, rqdatac.get_previous_trading_date(start_date), end_date, frequency='1d', fields='close').fillna(method='ffill').pct_change()[1:]

    market_cap = rqdatac.get_factor(id_or_symbols = market_list, factor = 'a_share_market_val', start_date = start_date, end_date = end_date)

    # 在传入的 Series/Dataframe 存在缺失值的情况下，pandas 的 dot 函数返回 nan，sum 函数剔除 nan 后返回求和结果。
    market_portfolio_daily_return = pd.Series(np.diag(market_daily_return.replace(np.nan, 0).dot((market_cap.replace(np.nan, 0)/market_cap.sum()).T)), index=daily_return.index)

    # 计算无风险日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=start_date, end_date=end_date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1).loc[daily_return.index]

    daily_excess_return = daily_return.T.subtract(risk_free_return.iloc[:,0]).T

    market_portfolio_daily_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:,0])

    # 剔除收益率数据少于66个的股票
    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > (len(daily_return) - 66)].index

    return daily_excess_return.drop(daily_excess_return[inds], axis=1), market_portfolio_daily_excess_return


def recent_annual_report(date):

    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

    # 取出最近一期财务报告类型，例如 '2016q3' 或  '2016q4'， 其中 '2016q3' 表示前三季度累计； '2016q4' 表示年报

    recent_report_type = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.income_statement.net_profit),
                                                  entry_date=date, interval='1y', report_quarter=True)['report_quarter']

    annual_report_type = recent_report_type.copy()  # 深拷贝

    # 若上市公司未发布今年的财报，且未发布去年的年报，则取前年的年报为最新年报

    if recent_report_type.T.iloc[0].values[0][:4] == str(previous_year):

        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year - 1) + 'q4'

    # 若上市公司已发布今年的财报，则取去年的年报为最新年报

    else:
        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year) + 'q4'

    # recent_report_type 和 annual_report_type 均为 dataframe 格式，输出时转为 Series 格式

    return recent_report_type.T[date], annual_report_type.T[date]


# 计算原生指标过去十二个月的滚动值（利润表、现金流量表滚动求和）
def ttm_sum(financial_indicator, date):

    recent_report_type, annual_report_type = recent_annual_report(date)
    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

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


def last_five_annual_report(date):
    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

    # 取出最近一期财务报告类型，例如 '2016q3' 或  '2016q4'， 其中 '2016q3' 表示前三季度累计； '2016q4' 表示年报

    recent_report_type = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.income_statement.net_profit),
                                                  entry_date=date, interval='1y', report_quarter=True)[
        'report_quarter']

    annual_report_type = recent_report_type.copy()  # 深拷贝

    # recent_report_type 和 annual_report_type 均为 dataframe 格式，输出时转为 Series 格式

    # 若上市公司未发布今年的财报，且未发布去年的年报，则取前年的年报为最新年报

    if recent_report_type.T.iloc[0].values[0][:4] == str(previous_year):

        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year - 1) + 'q4'

        # 若上市公司已发布今年的财报，则取去年的年报为最新年报

    else:
        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year) + 'q4'

    annual_report_type_last_year = annual_report_type.T[date].copy()
    annual_report_type_2_year_ago = annual_report_type.T[date].copy()
    annual_report_type_3_year_ago = annual_report_type.T[date].copy()
    annual_report_type_4_year_ago = annual_report_type.T[date].copy()

    for stock in annual_report_type.T.index.tolist():
        if annual_report_type.T[date][stock][:4] == str(previous_year):
            annual_report_type_last_year[stock] = str(previous_year - 1) + 'q4'
            annual_report_type_2_year_ago[stock] = str(previous_year - 2) + 'q4'
            annual_report_type_3_year_ago[stock] = str(previous_year - 3) + 'q4'
            annual_report_type_4_year_ago[stock] = str(previous_year - 4) + 'q4'
        else:
            annual_report_type_last_year[stock] = str(previous_year - 2) + 'q4'
            annual_report_type_2_year_ago[stock] = str(previous_year - 3) + 'q4'
            annual_report_type_3_year_ago[stock] = str(previous_year - 4) + 'q4'
            annual_report_type_4_year_ago[stock] = str(previous_year - 5) + 'q4'

    return recent_report_type.T[date], annual_report_type.T[date], \
           annual_report_type_last_year, annual_report_type_2_year_ago, annual_report_type_3_year_ago, annual_report_type_4_year_ago

