
import scipy.optimize as sc_opt


def winsorization_and_market_cap_weighed_standardization(factor_exposure, market_cap_on_current_day):

    # Winsorization

    upper_limit = factor_exposure.mean() + 3 * factor_exposure.std()

    lower_limit = factor_exposure.mean() - 3 * factor_exposure.std()

    # Replace the outliers

    factor_exposure[(factor_exposure > upper_limit) & (factor_exposure != np.nan)] = upper_limit

    factor_exposure[(factor_exposure < lower_limit) & (factor_exposure != np.nan)] = lower_limit

    # Market cap weighted standardized

    market_cap_weighted_mean = (market_cap_on_current_day * factor_exposure).sum() / market_cap_on_current_day.sum()

    return (factor_exposure - market_cap_weighted_mean) / factor_exposure.std()


def orthogonalize(target_variable, reference_variable, regression_weight):

    initial_guess = 1

    def objective_function(coef):

        return np.abs((regression_weight * (target_variable - coef * reference_variable) * reference_variable).sum())

    res = sc_opt.minimize(objective_function, x0=initial_guess, method='L-BFGS-B')

    orthogonalized_target_variable = target_variable - res['x'] * reference_variable

    return orthogonalized_target_variable

'''
def missing_data_treatment(factor_one,factor_two,factor_three,factor_one_part,factor_two_part,factor_three_part,market_cap_on_current_day):

    # 取出每个细分因子中空值对应的index，并求交集，所有因子均为空的情况不做缺失值处理
    factor_one_null = factor_one[factor_one.isnull()].index.tolist()
    factor_two_null = factor_two[factor_two.isnull()].index.tolist()
    factor_three_null = factor_three[factor_three.isnull()].index.tolist()

    # 取出所有因子都是空值对应的index
    factor_null = list((set(factor_one_null).intersection(set(factor_two_null))).intersection(set(factor_three_null)))

    # 对细分因子不全为nan的数据进行缺失值处理：将缺失的因子所占比按比例分摊至剩余非空因子
    factor_one_null_list = list(set(factor_one_null)-set(factor_null))
    factor_two_null_list = list(set(factor_two_null)-set(factor_null))
    factor_three_null_list = list(set(factor_three_null)-set(factor_null))

    factor_one_fillna = pd.Series(data=[
        factor_two[stock] if stock in factor_three_null else factor_three[stock] if stock in factor_two_null else
        factor_two[stock] * factor_two_part / (factor_two_part + factor_three_part) + factor_three[stock] * factor_three_part / (factor_two_part + factor_three_part)
        for stock in factor_one_null_list], index=factor_one_null_list)

    factor_two_fillna = pd.Series(data=[
        factor_one[stock] if stock in factor_three_null else factor_three[stock] if stock in factor_one_null else
        factor_one[stock] * factor_one_part / (factor_one_part + factor_three_part) + factor_three[stock] * factor_three_part / (factor_one_part + factor_three_part)
        for stock in factor_two_null_list],index=factor_two_null_list)

    factor_three_fillna = pd.Series(data=[
        factor_one[stock] if stock in factor_two_null else factor_two[stock] if stock in factor_one_null else
        factor_one[stock] * factor_one_part / (factor_one_part + factor_two_part) + factor_two[stock]*factor_two_part / (factor_one_part + factor_two_part)
        for stock in factor_three_null_list],index=factor_three_null_list)

    factor_one_list = list(set(factor_one.index)-set(factor_one_null_list))
    factor_two_list = list(set(factor_two.index)-set(factor_two_null_list))
    factor_three_list = list(set(factor_three.index)-set(factor_three_null_list))

    # 将处理后的缺失值和未经处理的因子数据合并
    factor_one_fill = pd.concat([factor_one[factor_one_list],factor_one_fillna],axis=0)
    factor_two_fill = pd.concat([factor_two[factor_two_list],factor_two_fillna],axis=0)
    factor_three_fill = pd.concat([factor_three[factor_three_list],factor_three_fillna],axis=0)

    # 将处理后的细分因子标准化
    standard_factor_one = winsorization_and_market_cap_weighed_standardization(factor_one_fill,market_cap_on_current_day)
    standard_factor_two = winsorization_and_market_cap_weighed_standardization(factor_two_fill,market_cap_on_current_day)
    standard_factor_three = winsorization_and_market_cap_weighed_standardization(factor_three_fill,market_cap_on_current_day)

    return standard_factor_one,standard_factor_two,standard_factor_three
'''