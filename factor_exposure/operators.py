
import scipy.optimize as sc_opt


def winsorization_and_market_cap_weighed_standardization(factor_exposure, market_cap_on_current_day):

    # standardized factor exposure use cap-weighted mean and equal-weighted standard deviation

    market_cap_weighted_mean = (market_cap_on_current_day * factor_exposure).sum() / market_cap_on_current_day.sum()

    standardized_factor_exposure = (factor_exposure - market_cap_weighted_mean) / factor_exposure.std()

    # Winsorization

    upper_limit = standardized_factor_exposure.mean() + 3 * standardized_factor_exposure.std()

    lower_limit = standardized_factor_exposure.mean() - 3 * standardized_factor_exposure.std()

    highest_point = standardized_factor_exposure.mean() + 10 * standardized_factor_exposure.std()

    lowest_point = standardized_factor_exposure.mean() - 10 * standardized_factor_exposure.std()

    # Replace & remove the outliers

    stock_list = standardized_factor_exposure.index.tolist()
    for stock in standardized_factor_exposure.index.tolist():
        if ((standardized_factor_exposure.loc[stock] > highest_point) | (standardized_factor_exposure.loc[stock] < lowest_point)) & (standardized_factor_exposure.loc[stock] != np.nan):
            stock_list.remove(stock)
    standardized_factor_exposure = standardized_factor_exposure[stock_list]

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

