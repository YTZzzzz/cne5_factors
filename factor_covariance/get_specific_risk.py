import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta



import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))


dailyParameters = {'factor_return_length': 252,
                    'sepcific_volatility_half_life': 42,
                    'Newey_West_Auto_Correlation_Lags': 0,
                    'Newey_West_Auto_correlation_half_life': np.nan,
                    'Bayesian_Shrinkage_parameter': 0.1,
                    'volatilityRegimeAdjustment_half_life': 4}

shortTermParameters = {'factor_return_length': 252,
                       'sepcific_volatility_half_life': 84,
                       'Newey_West_Auto_Correlation_Lags': 5,
                       'Newey_West_Auto_correlation_half_life': 252,
                       'Bayesian_Shrinkage_parameter': 0.25,
                       'volatilityRegimeAdjustment_half_life': 42}

longTermParameters = {'factor_return_length': 252,
                       'sepcific_volatility_half_life': 252,
                       'Newey_West_Auto_Correlation_Lags': 5,
                       'Newey_West_Auto_correlation_half_life': 252,
                       'Bayesian_Shrinkage_parameter': 0.25,
                       'volatilityRegimeAdjustment_half_life': 168}


def get_multiperiod_stock_returns(stock_list, latest_trading_date, parameters):

    # 取出多期的收益率，在 Newey West 中计算当期因子收益和滞后因子收益的经验协方差

    end_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=30), latest_trading_date, country='cn')[-parameters.get('Newey_West_Auto_Correlation_Lags'):]

    start_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=400), latest_trading_date, country='cn')[-(parameters.get('factor_return_length') + parameters.get('Newey_West_Auto_Correlation_Lags')):-parameters.get('factor_return_length')]

    # 以百分比为单位，所以乘以 100

    daily_specific_return = rqdatac.barra.get_specific_return(stock_list,start_dates[0],end_dates[-1])

    multiperiod_specific_return = {}

    for i in range(1, parameters.get('Newey_West_Auto_Correlation_Lags') + 1):

        multiperiod_specific_return['lag_' + str(i)] = daily_specific_return[-(parameters.get('factor_return_length') + i): -i]

    # 返回当期的因子收益序列，以及滞后N期的因子收益序列

    return daily_specific_return[-parameters.get('factor_return_length'):], multiperiod_specific_return


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))[::-1]


def Newey_West_adjustment(daily_specific_return, multiperiod_specific_return, parameters):

    # 计算经验协方差矩阵，同时进行年化处理（乘以 252）

    stock_list = daily_specific_return.columns.tolist()

    volatility_exp_weight = get_exponential_weight(parameters['sepcific_volatility_half_life'], parameters['factor_return_length'])

    correlation_exp_weight = get_exponential_weight(parameters['Newey_West_Auto_correlation_half_life'], parameters['factor_return_length'])

    demeaned_daily_specific_return = daily_specific_return - daily_specific_return.mean()

    exp_weighted_var = pd.Series(index=stock_list,data=0)

    estimated_var = pd.Series(index=stock_list,data=0)

    intermediate_var = pd.Series(index=stock_list,data=0)

    for stock in stock_list:

        estimated_var.loc[stock] = volatility_exp_weight.dot(demeaned_daily_specific_return[stock].values * demeaned_daily_specific_return[stock].values) / volatility_exp_weight.sum()

    for lag in range(1, 6):

        demeaned_lag_specific_return = multiperiod_specific_return['lag_' + str(lag)] - multiperiod_specific_return['lag_' + str(lag)].mean()

        for stock in stock_list:

            exp_weighted_var.loc[stock] = correlation_exp_weight.dot(demeaned_lag_specific_return[stock].values * demeaned_lag_specific_return[stock].values) / correlation_exp_weight.sum()

        intermediate_var = intermediate_var + (1-lag/(1+parameters.get('Newey_West_Auto_correlation_half_life')))*(exp_weighted_var + exp_weighted_var.T)

    Newey_West_adjustment_var = 252 * (estimated_var + intermediate_var)

    specific_risk_newey_west_adjuestment = Newey_West_adjustment_var * estimated_var

    # 计算该步骤调整风险矩阵各项 volatility和相关系数

    return specific_risk_newey_west_adjuestment


def structural_risk_adjustment(Newey_West_adjustment_var):

    return


def Bayesian_Shrinkage_adjustment():
    return

