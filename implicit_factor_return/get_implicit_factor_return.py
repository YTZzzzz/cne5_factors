import numpy as np
import pandas as pd
import statsmodels.api as st
from datetime import datetime
from datetime import timedelta

import rqdatac

#rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))

rqdatac.init('rice','rice',('192.168.10.64',16008))


def get_shenwan_industry_exposure(stock_list, date):

    industry_classification = rqdatac.shenwan_instrument_industry(stock_list, date)

    industry_classification_missing_stocks = list(set(stock_list) - set(industry_classification.index.tolist()))

    # 当传入股票过多时，对于缺失行业标记的股票，RQD 目前不会向前搜索，因此需要循环单个传入查找这些股票的行业标记

    if len(industry_classification_missing_stocks) != 0:

        print(date, 'industry_classification_missing_stocks', industry_classification_missing_stocks)

        for stock in industry_classification_missing_stocks:

            missing_industry_classification = rqdatac.shenwan_instrument_industry(stock, date)

            if missing_industry_classification != None:

                industry_classification = industry_classification.append(pd.Series([missing_industry_classification[0], missing_industry_classification[1]], index=['index_code','index_name'], name = stock))

    shenwan_industry_name = ['农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料', '纺织服装', '轻工制造',\
                             '医药生物', '公用事业', '交通运输', '房地产', '商业贸易', '休闲服务','综合', '建筑材料',  '建筑装饰', '电气设备',\
                             '国防军工', '计算机', '传媒', '通信', '银行', '非银金融', '汽车', '机械设备']


    # 在 stock_list 中仅有一个股票的情况下，返回格式为 tuple

    if isinstance(industry_classification, tuple):

        industry_name = industry_classification[1]

        industry_exposure_df = pd.DataFrame(0, index = shenwan_industry_name, columns = stock_list).T

        industry_exposure_df[industry_name] = 1

    else:

        industry_exposure_df = pd.DataFrame(0, index = industry_classification.index, columns = shenwan_industry_name)

        for industry in shenwan_industry_name:

            industry_exposure_df.loc[industry_classification[industry_classification['index_name'] == industry].index, industry] = 1

    return industry_exposure_df.index.tolist(), industry_exposure_df



def get_exposure(stock_list, date):

    latest_trading_date = str(rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

    non_missing_stock_list, industry_exposure = get_shenwan_industry_exposure(stock_list, date)

    style_exposure = rqdatac.get_style_factor_exposure(stock_list, date, date, factors = 'all')

    style_exposure.index = style_exposure.index.droplevel('date')

    factor_exposure = pd.concat([style_exposure, industry_exposure], axis = 1)

    factor_exposure['市场联动'] = 1

    return factor_exposure


def get_barra_exposure(stock_list, date):

    non_missing_stock_list, industry_exposure = get_shenwan_industry_exposure(stock_list, date)

    style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                     'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

    style_exposure = rqdatac.barra.get_factor_exposure(stock_list, date, date, style_factors)

    #style_exposure = rqdatac.get_style_factor_exposure(stock_list, date, date, factors = 'all')

    style_exposure.index = style_exposure.index.droplevel('date')

    factor_exposure = pd.concat([style_exposure, industry_exposure], axis = 1)

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

    factor_returns_values = np.dot(np.linalg.inv(complete_matrix), right_hand_side_vector.T)

    factor_returns = pd.Series(factor_returns_values[:-1], index = X.columns)

    return factor_returns



def factor_return_estimation(stock_list, date, factor_exposure):

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    previous_trading_date = rqdatac.get_previous_trading_date(latest_trading_date)

    # 计算无风险日收益率

    daily_return = rqdatac.get_price(order_book_ids=stock_list, start_date=previous_trading_date, end_date=latest_trading_date, fields='close').pct_change()[-1:].T

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=latest_trading_date, end_date=latest_trading_date, tenor='3M')['3M']

    daily_risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 252)) - 1)

    daily_excess_return = daily_return.subtract(daily_risk_free_return.values).T

    # 以市场平方根作为加权最小二乘法的加权系数

    market_cap = rqdatac.get_factor(id_or_symbols = stock_list, factor = 'a_share_market_val', start_date = latest_trading_date, end_date = latest_trading_date)

    normalized_regression_weight = market_cap.pow(0.5)/market_cap.pow(0.5).sum()

    # 各行业市值之和，用于行业收益率约束条件

    shenwan_industry_name = ['农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料', '纺织服装', '轻工制造',\
                             '医药生物', '公用事业', '交通运输', '房地产', '商业贸易', '休闲服务','综合', '建筑材料',  '建筑装饰', '电气设备',\
                             '国防军工', '计算机', '传媒', '通信', '银行', '非银金融', '汽车', '机械设备']

    industry_total_market_cap = market_cap.dot(factor_exposure[shenwan_industry_name] )

    factor_return_series = pd.DataFrame()

    # 对10个风格因子不添加约束，对 GICS 32个行业添加约束

    factor_return_series['whole_market'] = constrainted_weighted_least_square(Y = daily_excess_return[factor_exposure.index].values[0], X = factor_exposure, weight = normalized_regression_weight[factor_exposure.index],\
                                                                     industry_total_market_cap = industry_total_market_cap, unconstrained_variables = 10, constrained_variables = len(industry_total_market_cap))
    ### 沪深300

    csi_300_components = rqdatac.index_components(index_name = '000300.XSHG', date = previous_trading_date)

    # 各行业市值之和，用于行业收益率约束条件

    csi_300_industry_total_market_cap = market_cap[csi_300_components].dot(factor_exposure[shenwan_industry_name].loc[csi_300_components])

    # 若行业市值之和小于100，则认为基准没有配置该行业

    missing_industry = csi_300_industry_total_market_cap[csi_300_industry_total_market_cap < 100].index

    csi_300_industry_total_market_cap = csi_300_industry_total_market_cap.drop(missing_industry)

    factor_return_series['csi_300'] = constrainted_weighted_least_square(Y = daily_excess_return[factor_exposure.index][csi_300_components].values[0], X = factor_exposure.drop(missing_industry, axis =1).loc[csi_300_components], weight = normalized_regression_weight[factor_exposure.index][csi_300_components],\
                                                                industry_total_market_cap = csi_300_industry_total_market_cap, unconstrained_variables = 10, constrained_variables = len(csi_300_industry_total_market_cap))


    ### 中证500

    csi_500_components = rqdatac.index_components(index_name = '000905.XSHG', date = previous_trading_date)

    csi_500_industry_total_market_cap = market_cap[csi_500_components].dot(factor_exposure[shenwan_industry_name].loc[csi_500_components])

    missing_industry = csi_500_industry_total_market_cap[csi_500_industry_total_market_cap < 100].index

    csi_500_industry_total_market_cap = csi_500_industry_total_market_cap.drop(missing_industry)

    factor_return_series['csi_500'] = constrainted_weighted_least_square(Y = daily_excess_return[factor_exposure.index][csi_500_components].values[0], X = factor_exposure.drop(missing_industry, axis =1).loc[csi_500_components], weight = normalized_regression_weight[factor_exposure.index][csi_500_components],\
                                                                industry_total_market_cap = csi_500_industry_total_market_cap, unconstrained_variables = 10, constrained_variables = len(csi_500_industry_total_market_cap))


    ### 中证800

    csi_800_components = rqdatac.index_components(index_name = '000906.XSHG', date = previous_trading_date)

    csi_800_industry_total_market_cap = market_cap[csi_800_components].dot(factor_exposure[shenwan_industry_name].loc[csi_800_components])

    missing_industry = csi_800_industry_total_market_cap[csi_800_industry_total_market_cap < 100].index

    csi_800_industry_total_market_cap = csi_800_industry_total_market_cap.drop(missing_industry)

    factor_return_series['csi_800'] = constrainted_weighted_least_square(Y = daily_excess_return[factor_exposure.index][csi_800_components].values[0], X = factor_exposure.drop(missing_industry, axis =1).loc[csi_800_components], weight = normalized_regression_weight[factor_exposure.index][csi_800_components],\
                                                                industry_total_market_cap = csi_800_industry_total_market_cap, unconstrained_variables = 10, constrained_variables = len(csi_800_industry_total_market_cap))

    # 若指数在特定行业中没有配置任何股票，则因子收益率为 0

    return factor_return_series.replace(np.nan, 0)



def get_implicit_factor_return(date):

    latest_trading_date = str(rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

    previous_trading_date = str(rqdatac.get_previous_trading_date(latest_trading_date))

    # 取前一交易日全市场已经上市的股票，保证日收益率计算

    stock_list = rqdatac.all_instruments(type = 'CS', date = previous_trading_date)['order_book_id'].values.tolist()

    # 计算全市场前一交易日的行业暴露度

    #factor_exposure = get_exposure(stock_list, previous_trading_date)

    factor_exposure = get_barra_exposure(stock_list, date)

    factor_returns = factor_return_estimation(stock_list, date = date, factor_exposure= factor_exposure)

    return factor_returns





factor_returns = get_implicit_factor_return('2017-02-02')


industry_factors = ['CNE5S_ENERGY', 'CNE5S_CHEM', 'CNE5S_CONMAT', 'CNE5S_MTLMIN', 'CNE5S_MATERIAL', 'CNE5S_AERODEF', \
                    'CNE5S_BLDPROD', 'CNE5S_CNSTENG', 'CNE5S_ELECEQP', 'CNE5S_INDCONG', 'CNE5S_MACH', 'CNE5S_TRDDIST', \
                    'CNE5S_COMSERV', 'CNE5S_AIRLINE', 'CNE5S_MARINE', 'CNE5S_RDRLTRAN', 'CNE5S_AUTO', 'CNE5S_HOUSEDUR', \
                    'CNE5S_LEISLUX', 'CNE5S_CONSSERV', 'CNE5S_MEDIA', 'CNE5S_RETAIL', 'CNE5S_PERSPRD', 'CNE5S_BEV', \
                    'CNE5S_FOODPROD', 'CNE5S_HEALTH', 'CNE5S_BANKS', 'CNE5S_DVFININS', 'CNE5S_REALEST',
                    'CNE5S_SOFTWARE', 'CNE5S_HDWRSEMI', 'CNE5S_UTILITIE']

barra_style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                 'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

country_factor = ['CNE5S_COUNTRY']

all_factors = style_factors + country_factor + industry_factors

barra_factor_returns = rqdatac.barra.get_factor_return('2018-02-02', '2018-02-02', style_factors)

barra_index = pd.Series([1,2, 3, 4, 5, 6, 7, 8, 9, 10], index = barra_style_factors)

rq_style_factors = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility', 'growth',
                    'book_to_price', 'leverage', 'liquidity', 'non_linear_size']

rq_index = pd.Series(rq_style_factors, index = [1,2, 3, 4, 5, 6, 7, 8, 9, 10])

index_mapping = barra_index.map(rq_index)

barra_factor_returns.columns = rq_style_factors

barra_style_factor = pd.concat([barra_factor_returns.T, index_mapping], axis = 1)

barra_style_factor.columns = ['values', 'rq_index']

barra_style_factor.index = barra_style_factor['rq_index']

merged_factor_returns = pd.concat([factor_returns['whole_market'][rq_style_factors], barra_style_factor['values']], axis = 1)

merged_factor_returns.columns = ['replicated_factor_return', 'original_factor_return']

#merged_factor_returns.loc[industry_factors]

merged_factor_returns.loc[style_factors]

print('corr', merged_factor_returns.corr())




test_trading_dates = rqdatac.get_trading_dates('2017-01-01', '2018-04-02')
