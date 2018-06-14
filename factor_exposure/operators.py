import scipy.optimize as sc_opt
import numpy as np
import pandas as pd

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


def atomic_descriptors_imputation_and_combination(atomic_descriptors_df, atom_descriptors_weight):

    missing_desriptors_position_label = atomic_descriptors_df.notnull() + 0

    # 根据细分因子缺失位置，计算每一个股票的暴露度的归一化权重

    renormalized_weight = missing_desriptors_position_label.dot(atom_descriptors_weight)

    style_factor = atomic_descriptors_df.replace(np.nan, 0).dot(atom_descriptors_weight).divide(renormalized_weight)

    return style_factor


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

            if (industry_style_factor_exposure[factor].isnull().any() == False):

                continue

            else:

                missing_data_stock_list = industry_style_factor_exposure[factor].index[industry_style_factor_exposure[factor].astype(np.float).apply(np.isnan)]

                y = industry_style_factor_exposure[[factor, 'market_cap']].dropna()[factor].values

                x = np.stack([industry_style_factor_exposure[[factor, 'market_cap']].dropna()['market_cap'].values, np.ones(len(y))]).T

                st_result = st.OLS(y, x).fit()

                exogenous_variables = pd.concat([industry_style_factor_exposure['market_cap'][missing_data_stock_list], pd.Series(data = 1, index = missing_data_stock_list)], axis = 1)

                imputed_style_factors_exposure.loc[missing_data_stock_list, factor] = exogenous_variables.dot(st_result.params)

    return imputed_style_factors_exposure




