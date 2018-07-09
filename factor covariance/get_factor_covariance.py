import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta



import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))


dailyParameters = {'factor_return_length': 252,\
                    'volatility_half_life': 42,\
                    'NeweyWest_volatility_lags': np.nan,\
                    'correlation_half_life': 200,\
                    'NeweyWest_correlation_lags': np.nan,\
                    'volatilityRegimeAdjustment_half_life': 4}


shortTermParameters = {'factor_return_length': 252,\
                       'volatility_half_life': 84,\
                       'NeweyWest_volatility_lags': 5,\
                       'correlation_half_life': 504,\
                       'NeweyWest_correlation_lags': 2,\
                       'volatilityRegimeAdjustment_half_life': 42}

longTermParameters = {'factor_return_length': 252,\
                      'volatility_half_life': 252,\
                      'NeweyWest_volatility_lags': 5,\
                      'correlation_half_life': 504,\
                      'NeweyWest_correlation_lags': 2,\
                      'volatilityRegimeAdjustment_half_life': 168}



# 未进行 Eigenfactor risk adjustment 和 volatility regime adjustment

unadjusted_covariance = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_covariance/data/CNE5S_100_UnadjCovariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])

unadjusted_covariance = unadjusted_covariance.drop('DataDate', axis = 1)

# 已实现 Eigenfactor risk adjustment，未进行 volatility regime adjustment

pre_volatilityRegimeAdjustment_covariance = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_covariance/data/CNE5S_100_preVRACovariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])

# 已实现 Eigenfactor risk adjustment 和 volatility regime adjustment

fully_processed_covariance = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_covariance/data/CNE5S_100_Covariance.20180202.txt', sep='|', engine='python', header=0, skipfooter=1, skiprows=2, parse_dates=[])



def factor_covariance_comparison(reformatted_empirical_factor_covariance):

    # 未处理协方差矩阵

    merged_covariance = pd.merge(unadjusted_covariance, reformatted_empirical_factor_covariance, how='left', left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

    print('unadjusted comparison', merged_covariance[['VarCovar', 0]].corr())

    print('unadjusted comparison', merged_covariance[['VarCovar', 0]].head())


    # Eigenfactor risk adjustment 处理后

    merged_covariance = pd.merge(pre_volatilityRegimeAdjustment_covariance, reformatted_empirical_factor_covariance, how='left', left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

    print('pre_volatilityRegimeAdjustment comparison', merged_covariance[['VarCovar', 0]].corr())

    print('pre_volatilityRegimeAdjustment comparison', merged_covariance[['VarCovar', 0]].head())


    # volatility regime adjustment 处理后

    merged_covariance = pd.merge(fully_processed_covariance, reformatted_empirical_factor_covariance, how='left', left_on=['!Factor1', 'Factor2'], right_on=['factor', '_factor'])

    print('fully_processed comparison', merged_covariance[['VarCovar', 0]].corr())

    print('fully_processed comparison', merged_covariance[['VarCovar', 0]].head())




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

    # 计算经验协方差矩阵，同时进行年化处理（乘以 252）

    empirical_factor_covariance = current_factor_return.cov().stack() * 252

    empirical_factor_covariance.index.names = ['factor', '_factor']

    reformatted_empirical_factor_covariance = empirical_factor_covariance.reset_index()

    volatility_exp_weight = get_exponential_weight(parameters['volatility_half_life'], parameters['factor_return_length'])

    correlation_exp_weight = get_exponential_weight(parameters['correlation_half_life'], parameters['factor_return_length'])

    demeaned_current_factor_return = current_factor_return - current_factor_return.mean()

    for lag in range(1, 6):

        demeaned_lag_factor_return = multiperiod_factor_returns['lag_' + str(lag)] - multiperiod_factor_returns['lag_' + str(lag)].mean()

        for factor in all_factors:

            exp_weighted_cov = correlation_exp_weight.dot(demeaned_current_factor_return['CNE5S_BETA'].values * demeaned_lag_factor_return['CNE5S_BETA'].values)/correlation_exp_weight.sum()












    multiperiods_factor_returns = get_multiperiods_factor_returns(all_factors, latest_trading_date)


date = '2018-02-02'

def get_factor_covariance(date, parameters):

    industry_factors = ['CNE5S_ENERGY', 'CNE5S_CHEM', 'CNE5S_CONMAT', 'CNE5S_MTLMIN', 'CNE5S_MATERIAL','CNE5S_AERODEF',
                            'CNE5S_BLDPROD', 'CNE5S_CNSTENG', 'CNE5S_ELECEQP', 'CNE5S_INDCONG', 'CNE5S_MACH','CNE5S_TRDDIST',
                            'CNE5S_COMSERV', 'CNE5S_AIRLINE', 'CNE5S_MARINE', 'CNE5S_RDRLTRAN', 'CNE5S_AUTO','CNE5S_HOUSEDUR',
                            'CNE5S_LEISLUX', 'CNE5S_CONSSERV', 'CNE5S_MEDIA', 'CNE5S_RETAIL', 'CNE5S_PERSPRD', 'CNE5S_BEV',
                            'CNE5S_FOODPROD', 'CNE5S_HEALTH', 'CNE5S_BANKS', 'CNE5S_DVFININS', 'CNE5S_REALEST', 'CNE5S_SOFTWARE',
                            'CNE5S_HDWRSEMI', 'CNE5S_UTILITIE']


    style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                     'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

    country_factor = ['CNE5S_COUNTRY']

    all_factors = industry_factors + style_factors + country_factor

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    current_factor_return, multiperiod_factor_returns = get_multiperiod_factor_returns(all_factors, latest_trading_date, parameters)

    # 计算经验协方差矩阵，同时进行年化处理（乘以 252）

    empirical_factor_covariance = current_factor_return.cov().stack() * 252

    empirical_factor_covariance.index.names = ['factor', '_factor']

    reformatted_empirical_factor_covariance = empirical_factor_covariance.reset_index()
















    # 计算因子收益波动率

    factor_covariance = rqdatac.barra.get_factor_covariance(date)

    # 贝塔

    print('barra', np.sqrt(factor_covariance['CNE5S_BETA']['CNE5S_BETA']))

    print('empirical', (daily_factor_return.std() * np.sqrt(252) * 100)['CNE5S_BETA'])

    # 市值

    print('barra', np.sqrt(factor_covariance['CNE5S_SIZE']['CNE5S_SIZE']))

    print('empirical', (daily_factor_return.std() * np.sqrt(252) * 100)['CNE5S_SIZE'])

    # 残余波动率

    print('barra', np.sqrt(factor_covariance['CNE5S_RESVOL']['CNE5S_RESVOL']))

    print('empirical', (daily_factor_return.std() * np.sqrt(252) * 100)['CNE5S_RESVOL'])

    # 盈利率

    print('barra', np.sqrt(factor_covariance['CNE5S_EARNYILD']['CNE5S_EARNYILD']))

    print('empirical', (daily_factor_return.std() * np.sqrt(252) * 100)['CNE5S_EARNYILD'])

    # 流动性

    print('barra', np.sqrt(factor_covariance['CNE5S_LIQUIDTY']['CNE5S_LIQUIDTY']))

    print('empirical', (daily_factor_return.std() * np.sqrt(252) * 100)['CNE5S_LIQUIDTY'])



    # 银行

    print('barra', np.sqrt(factor_covariance['CNE5S_BANKS']['CNE5S_BANKS']))

    print('empirical', (daily_factor_return.std() * np.sqrt(252) * 100)['CNE5S_BANKS'])





















