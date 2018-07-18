import numpy as np
import pandas as pd
import statsmodels.api as st
from datetime import datetime
from datetime import timedelta
from functools import reduce


import rqdatac

rqdatac.init('rice', 'rice', ('192.168.10.64', 16030))


def get_shenwan_industry_exposure(stock_list, date):

    industry_classification = rqdatac.shenwan_instrument_industry(stock_list, date)

    if date > '2014-01-01':

        shenwan_industry_name = ['农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料', '纺织服装', '轻工制造',\
                                 '医药生物', '公用事业', '交通运输', '房地产', '商业贸易', '休闲服务','综合', '建筑材料',  '建筑装饰', '电气设备',\
                                 '国防军工', '计算机', '传媒', '通信', '银行', '非银金融', '汽车', '机械设备']
    else:

        shenwan_industry_name = ['金融服务', '房地产', '医药生物', '有色金属', '餐饮旅游', '综合', '建筑建材', '家用电器',
                                 '交运设备', '食品饮料', '电子', '信息设备', '交通运输', '轻工制造', '公用事业', '机械设备',
                                 '纺织服装', '农林牧渔', '商业贸易', '化工', '信息服务', '采掘', '黑色金属']

    industry_exposure_df = pd.DataFrame(0, index = industry_classification.index, columns = shenwan_industry_name)

    for industry in shenwan_industry_name:

        industry_exposure_df.loc[industry_classification[industry_classification['index_name'] == industry].index, industry] = 1

    return industry_exposure_df.index.tolist(), industry_exposure_df


def get_exposure(stock_list,date):

    non_missing_stock_list,industry_exposure = get_shenwan_industry_exposure(stock_list, date)

    style_exposure = rqdatac.get_style_factor_exposure(non_missing_stock_list, date, date, factors = 'all')

    style_exposure.index = style_exposure.index.droplevel('date')

    factor_exposure = pd.concat([style_exposure,industry_exposure],axis=1)

    factor_exposure['市场联动'] = 1

    return factor_exposure


def constrainted_weighted_least_square(Y, X, weight, industry_total_market_cap, unconstrained_variables, constrained_variables):

    # 直接求解线性方程组（推导参见 Bloomberg <China A Share Equity Fundamental Factor Model>）

    upper_left_block = 2*np.dot(X.T, np.dot(np.diag(weight), X))

    upper_right_block = np.append(np.append(np.zeros(unconstrained_variables), -industry_total_market_cap.values), np.zeros(1))

    upper_block = np.concatenate((upper_left_block, upper_right_block.reshape(unconstrained_variables + constrained_variables + 1, 1)), axis=1)

    lower_block = np.append(upper_right_block, 0)

    complete_matrix = np.concatenate((upper_block, lower_block.reshape(1, unconstrained_variables + constrained_variables + 2)), axis=0)

    right_hand_side_vector = np.append(2*np.dot(X.T, np.multiply(weight, Y)), 0)

    factor_returns_values = np.dot(np.linalg.inv(complete_matrix.astype(np.float)), right_hand_side_vector.T)

    factor_returns = pd.Series(factor_returns_values[:-1], index = X.columns)

    return factor_returns


def customized_factor_return_estimation(date, factor_exposure,stock_list):

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    previous_trading_date = rqdatac.get_previous_trading_date(latest_trading_date)

    # 计算无风险日收益率

    daily_return = rqdatac.get_price(order_book_ids=factor_exposure.index.tolist(), start_date=previous_trading_date, end_date=latest_trading_date, fields='close').pct_change()[-1:].T

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=latest_trading_date, end_date=latest_trading_date, tenor='3M')['3M']

    daily_risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1)

    daily_excess_return = daily_return.subtract(daily_risk_free_return.values).T

    # 以市场平方根作为加权最小二乘法的加权系数

    market_cap = rqdatac.get_factor(id_or_symbols = factor_exposure.index.tolist(), factor = 'a_share_market_val', start_date = previous_trading_date, end_date = previous_trading_date)

    missing_market_cap_stock = market_cap[market_cap.isnull()==True].index.tolist()

    if len(missing_market_cap_stock) > 0:

        price = rqdatac.get_price(missing_market_cap_stock,previous_trading_date,previous_trading_date,fields='close',frequency='1d').T

        shares = rqdatac.get_shares(missing_market_cap_stock,previous_trading_date,previous_trading_date,fields='total_a').T

        market_cap[market_cap.isnull() == True] = (price * shares)[previous_trading_date]

    normalized_regression_weight = market_cap.pow(0.5)/market_cap.pow(0.5).sum()

    # 各行业市值之和，用于行业收益率约束条件

    if str(previous_trading_date) > '2014-01-01':

        industry_factors = ['农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料', '纺织服装', '轻工制造',\
                            '医药生物', '公用事业', '交通运输', '房地产', '商业贸易', '休闲服务','综合', '建筑材料',  '建筑装饰', '电气设备',\
                            '国防军工', '计算机', '传媒', '通信', '银行', '非银金融', '汽车', '机械设备']
    else:

        industry_factors = ['金融服务', '房地产', '医药生物', '有色金属', '餐饮旅游', '综合', '建筑建材', '家用电器',
                            '交运设备', '食品饮料', '电子', '信息设备', '交通运输', '轻工制造', '公用事业', '机械设备',
                            '纺织服装', '农林牧渔', '商业贸易', '化工', '信息服务', '采掘', '黑色金属']

    stock_list = list(set(market_cap.index.tolist()).intersection(set(stock_list)))

    # 各行业市值之和，用于行业收益率约束条件

    customized_industry_total_market_cap = market_cap[stock_list].dot(factor_exposure[industry_factors].loc[stock_list])

    # 若行业市值之和小于100，则认为基准没有配置该行业

    missing_industry = customized_industry_total_market_cap[customized_industry_total_market_cap < 100].index

    csi_300_industry_total_market_cap = customized_industry_total_market_cap.drop(missing_industry)

    factor_return_series = constrainted_weighted_least_square(Y = daily_excess_return[factor_exposure.index][stock_list].values[0], X = factor_exposure.drop(missing_industry, axis =1).loc[stock_list], weight = normalized_regression_weight[factor_exposure.index][stock_list],\
                                                                industry_total_market_cap = csi_300_industry_total_market_cap, unconstrained_variables = 10, constrained_variables = len(csi_300_industry_total_market_cap))

    # 若指数在特定行业中没有配置任何股票，则因子收益率为 0

    return factor_return_series.replace(np.nan, 0)


def get_explicit_factor_returns(date,stock_list):
    """
    :param date:日期
    :return: pandas.Series
    """

    previous_trading_date = rqdatac.get_previous_trading_date(date)

    factor_exposures = rqdatac.get_style_factor_exposure(stock_list, previous_trading_date, previous_trading_date, "all").sort_index()

    factor_exposures.index=factor_exposures.index.droplevel(1)

    priceChange = rqdatac.get_price(stock_list, rqdatac.get_previous_trading_date(previous_trading_date),
                                   previous_trading_date, fields="close").pct_change().iloc[-1]

    def _calc_explicitReturns_with_stocksList(stocksList):
        # 根据股票池计算收益率
        _sizeBeta = factor_exposures[['size','beta']].loc[stocksList]

        _quantileGroup = _sizeBeta.apply(lambda x:pd.cut(x,bins=3,labels=False)+1).reset_index()
        _quantileStocks = _quantileGroup.groupby(['size','beta']).apply(lambda x:x.index.tolist())
        market_neutralize_stocks = _quantileStocks.apply(
            lambda x: pd.Series(stocksList).loc[x].values.tolist()).values.tolist()
        return factor_exposures.loc[stocksList].apply(lambda x,y=market_neutralize_stocks:_calc_single_explicit_returns(x,y))

    def _calc_single_explicit_returns(_factor_exposure,market_neutralize_stocks):
        # 计算单一因子收益率
        def _deuce(series):
            median = series.median()
            return [series[series<=median].index.tolist(),series[series>median].index.tolist()]

        deuceResults = np.array([_deuce(_factor_exposure[neutralized_stks]) for neutralized_stks in market_neutralize_stocks]).flatten()

        short_stocksList = list(reduce(lambda x,y:set(x)|set(y),np.array([s for i,s in enumerate(deuceResults) if i%2==0])))
        long_stockList = list(reduce(lambda x,y:set(x)|set(y),np.array([s for i,s in enumerate(deuceResults) if i%2==1])))

        return priceChange[long_stockList].mean() - priceChange[short_stocksList].mean()

    results = _calc_explicitReturns_with_stocksList(stock_list)

    return results


def get_customized_factor_return(date,universe,options,method):

    """

    PARAMETERS
    ----------
    date: str
         分析日期

    stock_list：list 用户指定的股票池


    options: dict 其他选择参数，

    包括：drop_st_stock: boolean, 是否剔除ST股 ; drop_new_stock: np.int 选择股票的上市日期限制（自然日）; drop_suspended_stock: boolean,是否剔除停牌股

    method: str default: implicit 可选"explicit" 用户选择计算因子收益率的方式


    RETURN
    ----------

    factor_return: Series, 依据用户指定的股票池计算出的因子收益率

    """

    latest_trading_date = str(rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

    previous_trading_date = str(rqdatac.get_previous_trading_date(latest_trading_date))

    # 依据用户的选择参数，对stock_list进行筛选


    # 若用户选择剔除ST股：

    if options.get('drop_st_stock') == True:

        is_st_df = rqdatac.is_st_stock(universe,start_date=date,end_date=date)

        is_st_df.index = is_st_df.index.astype(str)

        stock_list = is_st_df.loc[date][is_st_df.loc[date].values == False].index.tolist()

    # 若用户选择剔除停牌股：

    if options.get('drop_suspended_stock') == True:

        trading_volume = rqdatac.get_price(stock_list,start_date=date,end_date=date,frequency='1d',fields='volume',country='cn')

        stock_list = trading_volume.loc[date][trading_volume.loc[date].values > 0].index.tolist()

    # 根据用户输入的上市日期限制，剔除新股

    threshold = [latest_trading_date if options.get('drop_new_stock')==None else str(datetime.strptime(latest_trading_date, "%Y-%m-%d") - timedelta(days=options.get('drop_new_stock')))][0]

    stock_list = [stock for stock in stock_list if rqdatac.instruments(stock).listed_date <= threshold]

    # 计算指定股票池内股票前一交易日的行业暴露度

    factor_exposure = get_exposure(stock_list,str(previous_trading_date))

    # 根据上述暴露度计算因子收益率

    if method == 'implicit':

        factor_return = customized_factor_return_estimation(date,factor_exposure,stock_list)

    else:

        factor_return = get_explicit_factor_returns(date, stock_list)

    return factor_return

