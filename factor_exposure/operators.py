import scipy.optimize as sc_opt
import numpy as np
import pandas as pd
import statsmodels.api as st

import rqdatac

rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))

import sys

sys.path.append("/Users/rice/Documents/cne5_factors/factor_exposure/")

from intermediate_variables import *


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


def market_cap_imputation(stock_list,market_cap_on_current_day,latest_trading_date):

    missing_market_cap_list = list(set(stock_list) - set(market_cap_on_current_day.index.tolist()))

    price_on_current_day = rqdatac.get_price(missing_market_cap_list,start_date=latest_trading_date.strftime('%Y-%m-%d'),end_date=latest_trading_date.strftime('%Y-%m-%d'),frequency='1d',fields='close',adjust_type='none').T

    shares_on_current_day = rqdatac.get_shares(missing_market_cap_list,latest_trading_date.strftime('%Y-%m-%d'),latest_trading_date.strftime('%Y-%m-%d'),fields='total_a').T

    market_cap = pd.Series(data = (price_on_current_day * shares_on_current_day)[latest_trading_date.strftime('%Y-%m-%d')],index=missing_market_cap_list)

    if market_cap.isnull().any():

        missing_list = market_cap[market_cap.isnull()].index.tolist()

        trading_date_22_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=50), latest_trading_date, country='cn')[-22]

        missing_market_cap = (rqdatac.get_factor(id_or_symbols=missing_list, factor='a_share_market_val', start_date=trading_date_22_before.strftime('%Y-%m-%d'), end_date=latest_trading_date.strftime('%Y-%m-%d')).mean()).fillna(market_cap_on_current_day.mean())

        market_cap = pd.concat([market_cap,missing_market_cap])

    imputed_market_cap_on_current_day = pd.concat([market_cap_on_current_day,market_cap])

    return imputed_market_cap_on_current_day


def atomic_descriptors_imputation_and_combination(atomic_descriptors_df, atom_descriptors_weight):

    missing_desriptors_position_label = atomic_descriptors_df.notnull() + 0

    # 根据细分因子缺失位置，计算每一个股票的暴露度的归一化权重

    renormalized_weight = missing_desriptors_position_label.dot(atom_descriptors_weight)

    style_factor = atomic_descriptors_df.replace(np.nan, 0).dot(atom_descriptors_weight).divide(renormalized_weight)

    return style_factor


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


def style_factors_imputation(style_factors_exposure, market_cap_on_current_day, date):

    import statsmodels.api as st

    imputed_style_factors_exposure = style_factors_exposure.copy()

    industry_label = get_shenwan_industry_label(style_factors_exposure.index.tolist(), date)

    style_factors_exposure['market_cap'] = market_cap_on_current_day

    style_factors_exposure['industry_label'] = industry_label

    # 风格因子暴露度缺失值处理逻辑，是寻找同行业中市值类似的股票，然后以该股票的因子暴露度取值作为缺失股票的因子暴露度取值

    # 实现方法为在同行业的股票中，用没有缺失的股票因子暴露度对股票市值做回归（考虑截距项），得到相应的回归系数，再用出现因子暴露度缺失值的股票的市值乘以回归系数，得到股票因子暴露度缺失值的估计值。

    for industry in industry_label.unique():

        industry_style_factor_exposure = style_factors_exposure[style_factors_exposure['industry_label'] == industry]

        for factor in imputed_style_factors_exposure.columns:

            # 若某一行业中，股票对于特定因子的暴露度没有缺失，则跳过缺失值填补流程

            if industry_style_factor_exposure[factor].isnull().any() == False:

                continue

            else:

                missing_data_stock_list = industry_style_factor_exposure[factor].index[industry_style_factor_exposure[factor].astype(np.float).apply(np.isnan)]

                y = industry_style_factor_exposure[[factor, 'market_cap']].dropna()[factor].values

                x = np.stack([industry_style_factor_exposure[[factor, 'market_cap']].dropna()['market_cap'].values, np.ones(len(y))]).T

                st_result = st.OLS(y, x).fit()

                exogenous_variables = pd.concat([industry_style_factor_exposure['market_cap'][missing_data_stock_list], pd.Series(data = 1, index = missing_data_stock_list)], axis = 1)

                imputed_style_factors_exposure.loc[missing_data_stock_list, factor] = exogenous_variables.dot(st_result.params)

    return imputed_style_factors_exposure


def individual_factor_imputation(stock_list, factor, market_cap_on_current_day, date):

    industry_label = get_shenwan_industry_label(stock_list, date)

    merged_df = pd.concat([factor, market_cap_on_current_day, industry_label], axis = 1)

    merged_df.columns = ['factor', 'market_cap', 'industry_label']

    imputed_factor = merged_df['factor'].copy()

    # 和因子暴露度缺失值填补逻辑类似，以回归法填补因子的缺失值

    for industry in industry_label.unique():

        industry_merged_df = merged_df[merged_df['industry_label'] == industry]

        if industry_merged_df['factor'].isnull().any() == False:

            continue

        else:

            missing_value_stock_list = industry_merged_df['factor'].index[industry_merged_df['factor'].astype(np.float).apply(np.isnan)]

            y = industry_merged_df.dropna()['factor'].values

            x = np.stack([industry_merged_df.dropna()['market_cap'].values, np.ones(len(y))]).T

            st_result = st.OLS(y, x).fit()

            exogenous_variables = pd.concat([industry_merged_df['market_cap'][missing_value_stock_list], pd.Series(data = 1, index = missing_value_stock_list)], axis = 1)

            imputed_factor.loc[missing_value_stock_list] = exogenous_variables.dot(st_result.params)

    return imputed_factor


def factor_imputation(market_cap_on_current_day,style_factors_exposure):

    imputed_factor_exposure = style_factors_exposure.copy()

    style_factors_exposure['market_cap'] = market_cap_on_current_day

    for factor in style_factors_exposure.columns:

        if style_factors_exposure[factor].isnull().any() == False:

            continue

        else:

            missing_data_stock_list = style_factors_exposure[factor].index[style_factors_exposure[factor].astype(np.float).apply(np.isnan)]

            y = style_factors_exposure[[factor, 'market_cap']].dropna()[factor].values

            x = np.stack([style_factors_exposure[[factor, 'market_cap']].dropna()['market_cap'].values,
                          np.ones(len(y))]).T

            st_result = st.OLS(y, x).fit()

            exogenous_variables = pd.concat([style_factors_exposure['market_cap'][missing_data_stock_list],
                                             pd.Series(data=1, index=missing_data_stock_list)], axis=1)

            imputed_factor_exposure.loc[missing_data_stock_list, factor] = exogenous_variables.dot(st_result.params)

    return imputed_factor_exposure
