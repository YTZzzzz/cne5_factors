


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))[::-1]



def get_daily_excess_return(stock_list, start_date, end_date):

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    daily_return = rqdatac.get_price(stock_list, rqdatac.get_previous_trading_date(start_date), end_date, frequency='1d', fields='ClosingPx').fillna(method='ffill').pct_change()[1:]

    market_cap = rqdatac.get_factor(id_or_symbols = stock_list, factor = 'market_cap', start_date = start_date, end_date = end_date)

    # 在传入的 Series/Dataframe 存在缺失值的情况下，pandas 的 dot 函数返回 nan，sum 函数剔除 nan 后返回求和结果。

    market_portfolio_daily_return = pd.Series(np.diag(daily_return.replace(np.nan, 0).dot((market_cap.replace(np.nan, 0)/market_cap.sum()).T)), index=daily_return.index)

    # 计算无风险日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=start_date, end_date=end_date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1).loc[daily_return.index]

    daily_excess_return = daily_return.T.subtract(risk_free_return.iloc[:,0]).T

    market_portfolio_daily_excess_return = market_portfolio_daily_return.subtract(risk_free_return.iloc[:,0])

    # 剔除收益率数据少于66个的股票

    inds = daily_excess_return.isnull().sum()[daily_excess_return.isnull().sum() > 66].index

    return daily_excess_return.drop(daily_excess_return[inds], axis=1), market_portfolio_daily_excess_return












