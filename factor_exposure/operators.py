
import scipy.optimize as sc_opt
import numpy as np
import pandas as pd


# 剔除垃圾股
def drop_st_stock(stock_list,date):


    st_stock_list = rqdatac.is_st_stock(stock_list,start_date=date,end_date=date)

    stock_list_drop_st = [stock for stock in stock_list if st_stock_list.T.loc[stock].values[0] == False]

    return stock_list_drop_st




def winsorization_and_market_cap_weighed_standardization(factor_exposure, market_cap_on_current_day):

    # standardized factor exposure use cap-weighted mean and equal-weighted standard deviation

    market_cap_weighted_mean = (market_cap_on_current_day * factor_exposure).sum() / market_cap_on_current_day.sum()

    standardized_factor_exposure = (factor_exposure - market_cap_weighted_mean) / factor_exposure.std()

    # Winsorization

    upper_limit = standardized_factor_exposure.mean() + 3 * standardized_factor_exposure.std()

    lower_limit = standardized_factor_exposure.mean() - 3 * standardized_factor_exposure.std()

    standardized_factor_exposure[(standardized_factor_exposure > upper_limit) & (standardized_factor_exposure != np.nan)] = upper_limit

    standardized_factor_exposure[(standardized_factor_exposure < lower_limit) & (standardized_factor_exposure != np.nan)] = lower_limit

    return standardized_factor_exposure


def orthogonalize(target_variable, reference_variable, regression_weight):

    initial_guess = 1

    def objective_function(coef):

        return np.abs((regression_weight * (target_variable - coef * reference_variable) * reference_variable).sum())

    res = sc_opt.minimize(objective_function, x0=initial_guess, method='L-BFGS-B')

    orthogonalized_target_variable = target_variable - res['x'] * reference_variable

    return orthogonalized_target_variable



# 计算原生指标过去十二个月的滚动值（利润表、现金流量表滚动求和）

def get_ttm_sum(financial_indicator, date, recent_report_type, annual_report_type):

    previous_year = datetime.strptime(date, '%Y-%m-%d').year - 1

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

    recent_annual_values = recent_five_reports.loc[str(previous_year - 1) + 'q4']

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