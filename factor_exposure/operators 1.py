
import scipy.optimize as sc_opt


def winsorization_and_market_cap_weighed_standardization(factor_exposure, market_cap):

    # Winsorization

    upper_limit = factor_exposure.mean() + 3 * factor_exposure.std()

    lower_limit = factor_exposure.mean() - 3 * factor_exposure.std()

    # Replace the outliers

    factor_exposure[(factor_exposure > upper_limit) & (factor_exposure != np.nan)] = upper_limit

    factor_exposure[(factor_exposure < lower_limit) & (factor_exposure != np.nan)] = lower_limit

    # Market cap weighted standardized

    market_cap_weighted_mean = (market_cap * factor_exposure).sum() / market_cap.sum()

    return (factor_exposure - market_cap_weighted_mean) / factor_exposure.std()


def orthogonalize(target_variable, reference_variable, regression_weight):

    initial_guess = 1

    def objective_function(coef):

        return np.abs((regression_weight * (target_variable - coef * reference_variable) * reference_variable).sum())

    res = sc_opt.minimize(objective_function, x0=initial_guess, method='L-BFGS-B')

    orthogonalized_target_variable = target_variable - res['x'] * reference_variable

    return orthogonalized_target_variable

