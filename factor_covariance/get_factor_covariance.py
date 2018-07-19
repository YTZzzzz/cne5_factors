import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta



import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))


dailyParameters = {'factor_return_length': 252,
                    'volatility_half_life': 42,
                    'NeweyWest_volatility_lags': np.nan,
                    'correlation_half_life': 200,
                    'NeweyWest_correlation_lags': np.nan,
                    'volatilityRegimeAdjustment_half_life': 4}

shortTermParameters = {'factor_return_length': 252,
                       'volatility_half_life': 84,
                       'NeweyWest_volatility_lags': 5,
                       'correlation_half_life': 504,
                       'NeweyWest_correlation_lags': 2,
                       'volatilityRegimeAdjustment_half_life': 42}

longTermParameters = {'factor_return_length': 252,
                      'volatility_half_life': 252,
                      'NeweyWest_volatility_lags': 5,
                      'correlation_half_life': 504,
                      'NeweyWest_correlation_lags': 2,
                      'volatilityRegimeAdjustment_half_life': 168}


# 未进行 Eigenfactor risk adjustment 和 volatility regime adjustment

#unadjusted_covariance = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_covariance/data/CNE5S_100_UnadjCovariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])

unadjusted_covariance = pd.read_csv('/Users/rice/Desktop/covariance_data/CNE5S_100_UnadjCovariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])

unadjusted_covariance = unadjusted_covariance.drop('DataDate', axis = 1)

# 已实现 Eigenfactor risk adjustment，未进行 volatility regime adjustment

#pre_volatilityRegimeAdjustment_covariance = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_covariance/data/CNE5S_100_preVRACovariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])

pre_volatilityRegimeAdjustment_covariance = pd.read_csv('/Users/rice/Desktop/covariance_data/CNE5S_100_preVRACovariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])

# 已实现 Eigenfactor risk adjustment 和 volatility regime adjustment

#fully_processed_covariance = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_covariance/data/CNE5S_100_Covariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])

fully_processed_covariance = pd.read_csv('/Users/rice/Desktop/covariance_data/CNE5S_100_Covariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])


def factor_covariance_comparison(reformatted_empirical_factor_covariance):

    # 未处理协方差矩阵

    merged_covariance = pd.merge(unadjusted_covariance, reformatted_empirical_factor_covariance, how='left', left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

    print('unadjusted_covariance 欧几里得距离：', np.linalg.norm(merged_covariance['VarCovar']-merged_covariance[0]))

    print('unadjusted comparison', merged_covariance[['VarCovar', 0]].astype(np.float).corr())

    print('unadjusted comparison', merged_covariance[['VarCovar', 0]].head())

    # Eigenfactor risk adjustment 处理后

    merged_covariance = pd.merge(pre_volatilityRegimeAdjustment_covariance, reformatted_empirical_factor_covariance, how='left', left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

    print("pre_volatilityRegimeAdjustment 欧几里得距离：", np.linalg.norm(merged_covariance['VarCovar']-merged_covariance[0]))

    print('pre_volatilityRegimeAdjustment comparison', merged_covariance[['VarCovar', 0]].astype(np.float).corr())

    print('pre_volatilityRegimeAdjustment comparison', merged_covariance[['VarCovar', 0]].head())

    # volatility regime adjustment 处理后

    merged_covariance = pd.merge(fully_processed_covariance, reformatted_empirical_factor_covariance, how='left', left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

    print("fully_processed_covariance 欧几里得距离：", np.linalg.norm(merged_covariance['VarCovar']-merged_covariance[0]))

    print('fully_processed comparison', merged_covariance[['VarCovar', 0]].astype(np.float).corr())

    print('fully_processed comparison', merged_covariance[['VarCovar', 0]].head())

    return


def get_multiperiod_factor_returns(all_factors, latest_trading_date, parameters):

    # 取出多期的收益率，在 Newey West 中计算当期因子收益和滞后因子收益的经验协方差

    end_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=30), latest_trading_date, country='cn')[-parameters.get('NeweyWest_volatility_lags'):]

    start_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=400), latest_trading_date, country='cn')[-(parameters.get('factor_return_length') + parameters.get('NeweyWest_volatility_lags')):-parameters.get('factor_return_length')]

    # 以百分比为单位，所以乘以 100

    daily_factor_return = rqdatac.barra.get_factor_return(start_dates[0], end_dates[-1], all_factors) * 100

    multiperiod_daily_factor_return = {}

    for i in range(1, parameters.get('NeweyWest_volatility_lags') + 1):

        multiperiod_daily_factor_return['lag_' + str(i)] = daily_factor_return[-(parameters.get('factor_return_length') + i): -i]

    # 返回当期的因子收益序列，以及滞后N期的因子收益序列

    return daily_factor_return[-parameters.get('factor_return_length'):], multiperiod_daily_factor_return


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))[::-1]


def Newey_West_adjustment(current_factor_return, multiperiod_factor_returns, all_factors, parameters):

    volatility_exp_weight = get_exponential_weight(parameters['volatility_half_life'], parameters['factor_return_length'])

    correlation_exp_weight = get_exponential_weight(parameters['correlation_half_life'], parameters['factor_return_length'])

    demeaned_current_factor_return = current_factor_return - current_factor_return.mean()

    estimated_lag_cov = pd.DataFrame(index=all_factors, columns=all_factors)

    estimated_lag_var = pd.DataFrame(index=all_factors, columns=all_factors)

    exp_factors_return = pd.DataFrame()

    varexp_factors_return = pd.DataFrame()

    intermediate_cov = pd.DataFrame(index=all_factors, columns=all_factors, data=0)

    intermediate_var = pd.DataFrame(index=all_factors, columns=all_factors, data=0)

    # 计算协方差和方差估计矩阵

    for factor in all_factors:

        exp_factors_return[factor] = np.sqrt(correlation_exp_weight) * demeaned_current_factor_return[factor] / correlation_exp_weight.sum()

        varexp_factors_return[factor] = np.sqrt(volatility_exp_weight) * demeaned_current_factor_return[factor] / volatility_exp_weight.sum()

    estimated_cov = exp_factors_return.cov()

    estimated_var = varexp_factors_return.cov()

    # 计算协方差和方差在不同滞后长度下的协方差和方差矩阵

    for lag in range(1, 6):

        demeaned_lag_factor_return = multiperiod_factor_returns['lag_' + str(lag)] - multiperiod_factor_returns['lag_' + str(lag)].mean()

        for factor in all_factors:

            estimated_lag_cov[factor] = np.sqrt(correlation_exp_weight).dot((exp_factors_return[factor].values * demeaned_lag_factor_return[factor]).values)

            estimated_lag_var[factor] = np.sqrt(volatility_exp_weight).dot((varexp_factors_return[factor].values * demeaned_lag_factor_return[factor]).values)

        if lag <= parameters['NeweyWest_correlation_lags']:

            intermediate_cov = intermediate_cov + (1 - lag / (1 + parameters.get('NeweyWest_correlation_lags'))) * (estimated_lag_cov + estimated_lag_cov.T)

        intermediate_var = intermediate_var + (1 - lag / (1 + parameters.get('NeweyWest_volatility_lags'))) * (estimated_lag_var + estimated_lag_var.T).replace(np.nan,0)

    Newey_West_adjustment_cov = 252 * (estimated_cov + intermediate_cov)

    Newey_West_adjustment_var = 252 * (estimated_var + intermediate_var)

    # 计算调整风险矩阵各项 volatility和相关系数

    correlation_matrix = pd.DataFrame(index=all_factors,columns=all_factors)

    factor_volitality_cov = pd.Series(data=np.sqrt(np.diag(Newey_West_adjustment_cov.astype(np.float))),index=Newey_West_adjustment_cov.index)

    factor_volitality = pd.Series(data=np.sqrt(np.diag(Newey_West_adjustment_var.astype(np.float))),index=Newey_West_adjustment_var.index)

    adjusted_covariance = pd.DataFrame(index=all_factors,columns=all_factors)

    for factor in all_factors:

        for factors in all_factors:

            correlation_matrix[factor].loc[factors] = Newey_West_adjustment_cov[factor][factors]/(factor_volitality_cov.loc[factors] * factor_volitality_cov.loc[factor])

            adjusted_covariance[factor][factors] = correlation_matrix[factor].loc[factors] * factor_volitality.loc[factor] * factor_volitality.loc[factors]

    return adjusted_covariance,factor_volitality,correlation_matrix,estimated_cov


def eigenfactor_risk_adjustment(Newey_West_adjustment_cov,factor_volitality,all_factors,estimated_cov):

    eigen_value, eigen_vector = np.linalg.eig(Newey_West_adjustment_cov.astype(np.float))

    eigen_value_matrix = pd.DataFrame(index=Newey_West_adjustment_cov.index, columns=Newey_West_adjustment_cov.index,data=np.diag(eigen_value.reshape(1, len(Newey_West_adjustment_cov.index))[0]))

    eigen_vector_matrix = pd.DataFrame(data=eigen_vector,index=Newey_West_adjustment_cov.index,columns=Newey_West_adjustment_cov.index)

    monte_carlo_sampling_number = 10000

    intermediate_eigen_value = pd.DataFrame(data=0,index=all_factors,columns=all_factors)

    for m in range(1,monte_carlo_sampling_number):

        monte_carlo_simulation = pd.DataFrame(columns=all_factors)

        for factor in all_factors:

            monte_carlo_simulation[factor] = np.random.normal(0, factor_volitality.loc[factor], size=252)

        monte_carlo_cov = (monte_carlo_simulation.dot(eigen_vector_matrix)).cov()

        eigen_value_m, eigen_vector_m = np.linalg.eig(monte_carlo_cov.astype(np.float))

        eigen_value_m_matrix = pd.DataFrame(index=monte_carlo_cov.index,columns=monte_carlo_cov.index,data=np.diag(eigen_value_m.reshape(1, len(monte_carlo_cov.index))[0]))

        eigen_value_adjust, eigen_vector_adjust = np.linalg.eig((monte_carlo_cov+estimated_cov).astype(np.float))

        eigen_value_adjust_matrix = pd.DataFrame(index=monte_carlo_cov.index,columns=monte_carlo_cov.index,data=np.diag(eigen_value_adjust.reshape(1, len(monte_carlo_cov.index))[0]))

        intermediate_eigen_value = intermediate_eigen_value + (eigen_value_adjust_matrix/eigen_value_m_matrix).replace(np.nan,0)

    pai = intermediate_eigen_value/monte_carlo_sampling_number

    eigenfactor_risk_adjustment_cov = eigen_vector_matrix.dot(pai.dot(eigen_value_matrix).dot(eigen_vector_matrix.T))

    return eigenfactor_risk_adjustment_cov


def volatility_regime_adjustment(eigenfactor_risk_adjustment_cov,current_factor_return,parameters):

    volatility_regime_exp_weight = get_exponential_weight(parameters['volatilityRegimeAdjustment_half_life'], parameters['factor_return_length'])

    empirical_factor_volitality = pd.Series(data=np.sqrt(np.diag(current_factor_return.cov())),index=current_factor_return.columns)

    bias = pd.Series(index=current_factor_return.index)

    for date in current_factor_return.index.tolist():

        bias.loc[date] = np.square(current_factor_return.loc[date]/empirical_factor_volitality).sum()/len(current_factor_return.columns)

    lambda_f = np.sqrt(volatility_regime_exp_weight.dot(bias)/volatility_regime_exp_weight.sum())

    volatility_regime_adjustment_cov = lambda_f**(2) * eigenfactor_risk_adjustment_cov

    return volatility_regime_adjustment_cov


def get_factor_covariance(date, parameters):
    industry_factors = ['CNE5S_ENERGY', 'CNE5S_CHEM', 'CNE5S_CONMAT', 'CNE5S_MTLMIN', 'CNE5S_MATERIAL', 'CNE5S_AERODEF',
                        'CNE5S_BLDPROD', 'CNE5S_CNSTENG', 'CNE5S_ELECEQP', 'CNE5S_INDCONG', 'CNE5S_MACH',
                        'CNE5S_TRDDIST',
                        'CNE5S_COMSERV', 'CNE5S_AIRLINE', 'CNE5S_MARINE', 'CNE5S_RDRLTRAN', 'CNE5S_AUTO',
                        'CNE5S_HOUSEDUR',
                        'CNE5S_LEISLUX', 'CNE5S_CONSSERV', 'CNE5S_MEDIA', 'CNE5S_RETAIL', 'CNE5S_PERSPRD', 'CNE5S_BEV',
                        'CNE5S_FOODPROD', 'CNE5S_HEALTH', 'CNE5S_BANKS', 'CNE5S_DVFININS', 'CNE5S_REALEST',
                        'CNE5S_SOFTWARE',
                        'CNE5S_HDWRSEMI', 'CNE5S_UTILITIE']

    style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                     'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

    country_factor = ['CNE5S_COUNTRY']

    all_factors = industry_factors + style_factors + country_factor

    latest_trading_date = rqdatac.get_previous_trading_date((datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

    current_factor_return, multiperiod_factor_returns = get_multiperiod_factor_returns(all_factors, latest_trading_date,
                                                                                       parameters)

    # 计算经验协方差矩阵，同时进行年化处理（乘以 252）

    empirical_factor_covariance = current_factor_return.cov().stack() * 252

    empirical_factor_covariance.index.names = ['factor', '_factor']

    reformatted_empirical_factor_covariance = empirical_factor_covariance.reset_index()

    Newey_West_adjustment_cov, factor_volitality, correlation_matrix,estimated_cov = Newey_West_adjustment(current_factor_return,multiperiod_factor_returns,all_factors, parameters)

    eigenfactor_risk_adjustment_cov = eigenfactor_risk_adjustment(Newey_West_adjustment_cov, factor_volitality,
                                                                  all_factors)

    volatility_regime_adjustment_cov = volatility_regime_adjustment(eigenfactor_risk_adjustment_cov,
                                                                    current_factor_return, parameters)

    return volatility_regime_adjustment_cov


date = '2018-02-02'

industry_factors = ['CNE5S_ENERGY', 'CNE5S_CHEM', 'CNE5S_CONMAT', 'CNE5S_MTLMIN', 'CNE5S_MATERIAL', 'CNE5S_AERODEF',
                    'CNE5S_BLDPROD', 'CNE5S_CNSTENG', 'CNE5S_ELECEQP', 'CNE5S_INDCONG', 'CNE5S_MACH', 'CNE5S_TRDDIST',
                    'CNE5S_COMSERV', 'CNE5S_AIRLINE', 'CNE5S_MARINE', 'CNE5S_RDRLTRAN', 'CNE5S_AUTO', 'CNE5S_HOUSEDUR',
                    'CNE5S_LEISLUX', 'CNE5S_CONSSERV', 'CNE5S_MEDIA', 'CNE5S_RETAIL', 'CNE5S_PERSPRD', 'CNE5S_BEV',
                    'CNE5S_FOODPROD', 'CNE5S_HEALTH', 'CNE5S_BANKS', 'CNE5S_DVFININS', 'CNE5S_REALEST',
                    'CNE5S_SOFTWARE','CNE5S_HDWRSEMI', 'CNE5S_UTILITIE']

style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                 'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

country_factor = ['CNE5S_COUNTRY']

all_factors = industry_factors + style_factors + country_factor

latest_trading_date = rqdatac.get_previous_trading_date((datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

current_factor_return, multiperiod_factor_returns = get_multiperiod_factor_returns(all_factors, latest_trading_date,shortTermParameters)

unadjusted_covariance_df = pd.DataFrame(index=all_factors,columns=all_factors)

pre_volatility_covariance_df = pd.DataFrame(index=all_factors,columns=all_factors)

fully_processed_covariance_df = pd.DataFrame(index=all_factors,columns=all_factors)

for factor in all_factors:
    for factors in all_factors:

        value = unadjusted_covariance[(unadjusted_covariance['!Factor1']==factor) &(unadjusted_covariance['Factor2']==factors)]['VarCovar'].values

        if len(value) == 0:
            unadjusted_covariance_df.loc[factor][factors] = unadjusted_covariance[(unadjusted_covariance['!Factor1']==factors) &(unadjusted_covariance['Factor2']==factor)]['VarCovar'].values[0]
            pre_volatility_covariance_df.loc[factor][factors] = pre_volatilityRegimeAdjustment_covariance[(pre_volatilityRegimeAdjustment_covariance['!Factor1'] == factors) & (pre_volatilityRegimeAdjustment_covariance['Factor2'] == factor)]['VarCovar'].values[0]
            fully_processed_covariance_df.loc[factor][factors] = fully_processed_covariance[(fully_processed_covariance['!Factor1'] == factors) & (fully_processed_covariance['Factor2'] == factor)]['VarCovar'].values[0]
        else:
            unadjusted_covariance_df.loc[factor][factors] = value[0]
            pre_volatility_covariance_df.loc[factor][factors] = pre_volatilityRegimeAdjustment_covariance[(pre_volatilityRegimeAdjustment_covariance['!Factor1'] == factor) & (pre_volatilityRegimeAdjustment_covariance['Factor2'] == factors)]['VarCovar'].values[0]
            fully_processed_covariance_df.loc[factor][factors] = fully_processed_covariance[(fully_processed_covariance['!Factor1'] == factor) & (fully_processed_covariance['Factor2'] == factors)]['VarCovar'].values[0]


factor_volitality1 = pd.Series(data=np.sqrt(np.diag(unadjusted_covariance_df.astype(np.float))),index=unadjusted_covariance_df.index)

Newey_West_adjustment_cov, factor_volitality, correlation_matrix,estimated_cov = Newey_West_adjustment(current_factor_return, multiperiod_factor_returns, all_factors, shortTermParameters)

eigenfactor_risk_adjustment_cov = eigenfactor_risk_adjustment(Newey_West_adjustment_cov, factor_volitality, all_factors,estimated_cov)

volatility_regime_adjustment_cov = volatility_regime_adjustment(pre_volatility_covariance_df,current_factor_return,shortTermParameters)


Newey_West_adjustment_cov = Newey_West_adjustment_cov.stack()

Newey_West_adjustment_cov.index.names = ['factor', '_factor']

Newey_West_adjustment_cov = Newey_West_adjustment_cov.reset_index()

eigenfactor_risk_adjustment_cov = eigenfactor_risk_adjustment_cov.stack()

eigenfactor_risk_adjustment_cov.index.names = ['factor', '_factor']

eigenfactor_risk_adjustment_cov = eigenfactor_risk_adjustment_cov.reset_index()

unadjusted_covariance_df = unadjusted_covariance_df.stack()

unadjusted_covariance_df.index.names = ['factor', '_factor']

unadjusted_covariance_df = unadjusted_covariance_df.reset_index()


# volatility regime adjustment 处理后

merged_covariance1 = pd.merge(pre_volatilityRegimeAdjustment_covariance, unadjusted_covariance_df, how='left',
                             left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

print('fully_processed comparison', merged_covariance1[['VarCovar', 0]].astype(np.float).corr())

print('fully_processed comparison', merged_covariance1[['VarCovar', 0]].head())

np.linalg.norm(merged_covariance1['VarCovar'] - merged_covariance1[0])


merged_covariance = pd.merge(pre_volatilityRegimeAdjustment_covariance, eigenfactor_risk_adjustment_cov, how='left',
                             left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

print('fully_processed comparison', merged_covariance[['VarCovar', 0]].astype(np.float).corr())

print('fully_processed comparison', merged_covariance[['VarCovar', 0]].head())

np.linalg.norm(merged_covariance['VarCovar'] - merged_covariance[0])



pre_volatility_covariance_df = pre_volatility_covariance_df.stack()

pre_volatility_covariance_df.index.names = ['factor', '_factor']

pre_volatility_covariance_df = pre_volatility_covariance_df.reset_index()

volatility_regime_adjustment_cov = volatility_regime_adjustment_cov.stack()

volatility_regime_adjustment_cov.index.names = ['factor', '_factor']

volatility_regime_adjustment_cov = volatility_regime_adjustment_cov.reset_index()

merged_covariance3 = pd.merge(fully_processed_covariance, pre_volatility_covariance_df, how='left',
                             left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

print('fully_processed comparison', merged_covariance3[['VarCovar', 0]].astype(np.float).corr())

print('fully_processed comparison', merged_covariance3[['VarCovar', 0]].head())

np.linalg.norm(merged_covariance3['VarCovar'] - merged_covariance3[0])


merged_covariance = pd.merge(fully_processed_covariance, volatility_regime_adjustment_cov, how='left',
                             left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

print('fully_processed comparison', merged_covariance[['VarCovar', 0]].astype(np.float).corr())

print('fully_processed comparison', merged_covariance[['VarCovar', 0]].head())

np.linalg.norm(merged_covariance['VarCovar'] - merged_covariance[0])


















