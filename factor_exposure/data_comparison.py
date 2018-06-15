import numpy as np
import pandas as pd
import statsmodels.api as st
from datetime import datetime
from datetime import timedelta
import scipy.optimize as sc_opt
import pickle

import matplotlib.pyplot as plt
from matplotlib import font_manager

import rqdatac

#rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))

rqdatac.init('rice','rice',('192.168.10.64',16008))


def get_style_exposure(stock_list, date):

    latest_trading_date = str(rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

    style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                     'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

    style_factor_exposure = rqdatac.barra.get_factor_exposure(stock_list, latest_trading_date, latest_trading_date, style_factors)

    style_factor_exposure.index = style_factor_exposure.index.droplevel('date')

    return style_factor_exposure


def get_barra_style_exposure(date):

    latest_trading_date = str(rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

    stock_list = rqdatac.all_instruments(type = 'CS', date = latest_trading_date)['order_book_id'].values.tolist()


    style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                     'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

    style_factor_exposure = rqdatac.barra.get_factor_exposure(stock_list, latest_trading_date, latest_trading_date, style_factors)

    style_factor_exposure.index = style_factor_exposure.index.droplevel('date')

    return style_factor_exposure


# 比对

date = '2017-01-06'

latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))
stock_list = rqdatac.all_instruments(type = 'CS', date = latest_trading_date)['order_book_id'].values.tolist()

atomic_descriptors_exposure, style_factors_exposure, imputed_style_factors_exposure,stock_beta = get_style_factors(date)

barra_style_factor_exposure = get_barra_style_exposure(date)

# earnings_yield

imputed_earnings_yield_correlation = pd.concat([imputed_style_factors_exposure['earnings_yield'].astype(np.float), barra_style_factor_exposure['CNE5S_EARNYILD']], axis =1).corr()

imputed_beta_correlation = pd.concat([imputed_style_factors_exposure['beta'], barra_style_factor_exposure['CNE5S_BETA']], axis =1).corr()

imputed_momentum_correlation = pd.concat([imputed_style_factors_exposure['momentum'], barra_style_factor_exposure['CNE5S_MOMENTUM']], axis =1).corr()

imputed_size_correlation = pd.concat([imputed_style_factors_exposure['size'], barra_style_factor_exposure['CNE5S_SIZE']], axis =1).corr()

imputed_resvol_correlation = pd.concat([imputed_style_factors_exposure['residual_volatility'], barra_style_factor_exposure['CNE5S_RESVOL']], axis =1).corr()

imputed_growth_correlation = pd.concat([imputed_style_factors_exposure['growth'], barra_style_factor_exposure['CNE5S_GROWTH']], axis =1).corr()

imputed_book_to_price_correlation = pd.concat([imputed_style_factors_exposure['book_to_price'].astype(np.float), barra_style_factor_exposure['CNE5S_BTOP']], axis =1).corr()

imputed_leverage_correlation = pd.concat([imputed_style_factors_exposure['leverage'], barra_style_factor_exposure['CNE5S_LEVERAGE']], axis =1).corr()

imputed_liquidity_correlation = pd.concat([imputed_style_factors_exposure['liquidity'], barra_style_factor_exposure['CNE5S_LIQUIDTY']], axis =1).corr()

imputed_non_linear_size_correlation = pd.concat([imputed_style_factors_exposure['non_linear_size'], barra_style_factor_exposure['CNE5S_SIZENL']], axis =1).corr()




liquidity_correlation = pd.concat([atomic_descriptors_exposure['earnings_growth'].astype(np.float), barra_style_factor_exposure['CNE5S_GROWTH']], axis =1).dropna().corr()



earnings_yield_correlation = pd.concat([imputed_style_factors_exposure['earnings_yield'].dropna(), barra_style_factor_exposure['CNE5S_EARNYILD'].loc[style_factors_exposure['earnings_yield'].dropna().index]], axis =1).corr()

imputed_growth_correlation = pd.concat([imputed_style_factors_exposure['growth'], barra_style_factor_exposure['CNE5S_GROWTH']], axis =1).corr()

growth_correlation = pd.concat([imputed_style_factors_exposure['earnings_yield'].dropna(), barra_style_factor_exposure['CNE5S_EARNYILD'].loc[style_factors_exposure['earnings_yield'].dropna().index]], axis =1).corr()

# 市值因子比对

date = '2018-04-27'

latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

previous_trading_date = rqdatac.get_previous_trading_date(date)

stock_list = rqdatac.all_instruments(type = 'CS', date = latest_trading_date)['order_book_id'].values.tolist()

#market_cap_on_current_day = rqdatac.get_factor(id_or_symbols = stock_list, factor = 'a_share_market_val', start_date = latest_trading_date.strftime('%Y-%m-%d'), end_date = latest_trading_date.strftime('%Y-%m-%d'))

#market_cap_on_current_day = rqdatac.get_factor(id_or_symbols = stock_list, factor = 'a_share_market_val_2', start_date = latest_trading_date.strftime('%Y-%m-%d'), end_date = latest_trading_date.strftime('%Y-%m-%d'))

#market_cap_on_current_day = rqdatac.get_factor(id_or_symbols = stock_list, factor = 'market_cap', start_date = latest_trading_date.strftime('%Y-%m-%d'), end_date = latest_trading_date.strftime('%Y-%m-%d'))

#market_cap_on_current_day = rqdatac.get_factor(id_or_symbols = stock_list, factor = 'market_cap_2', start_date = latest_trading_date.strftime('%Y-%m-%d'), end_date = latest_trading_date.strftime('%Y-%m-%d'))

market_cap_on_current_day = rqdatac.get_factor(id_or_symbols = stock_list, factor = 'a_share_market_val', start_date = previous_trading_date.strftime('%Y-%m-%d'), end_date = previous_trading_date.strftime('%Y-%m-%d'))


size_exposure = size(market_cap_on_current_day)

merged_size_expousre = pd.concat([size_exposure, style_factor_exposure['CNE5S_SIZE']], axis = 1)

print('min', merged_size_expousre.min())

print('max', merged_size_expousre.max())

print('corr', merged_size_expousre.corr())

print('tail', merged_size_expousre.tail())


# 市值因子比对

date = '2018-04-27'

factor_data = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_exposure/data/CNE5S_LOCALID_180427.RSK.csv')

market_cap_on_current_day = pd.Series(data = factor_data['CAPITALIZATION'].values, index = factor_data['order_book_id'].values.tolist())

a_share_market_val = rqdatac.get_factor(id_or_symbols=factor_data['order_book_id'].values.tolist(), factor='a_share_market_val', start_date = date, end_date = date)

size_exposure = size(market_cap_on_current_day)

merged_market_cap = pd.concat([market_cap_on_current_day, a_share_market_val], axis = 1)

merged_size_exposure = pd.concat([size_exposure, pd.Series(factor_data['SIZE'].values, index = factor_data['order_book_id'].values.tolist())], axis = 1)

print('min', merged_size_expousre.min())

print('max', merged_size_expousre.max())

print('corr', merged_size_expousre.corr())

print('tail', merged_size_expousre.tail())



market_cap = rqdatac.get_factor(id_or_symbols=factor_data['order_book_id'].values.tolist(), factor='market_cap', start_date = date, end_date = date)

st_stocks_filtered_stock_list = drop_st_stock(factor_data['order_book_id'].values.tolist(),date)

#st_stocks_filtered_stock_list = drop_suspended_stock(st_stocks_filtered_stock_list, date)

st_filtered_market_cap = market_cap_on_current_day.loc[st_stocks_filtered_stock_list]

test_market_cap = st_filtered_market_cap.sort_values()[2000:]

size_exposure = winsorization_and_market_cap_weighed_standardization(np.log(test_market_cap), test_market_cap)

size_exposure.mean()

normalized_market_cap_on_current_day = market_cap_on_current_day/market_cap_on_current_day.sum()

market_cap_weighted_mean = (market_cap_on_current_day * factor_exposure).sum() / market_cap_on_current_day.sum()

(market_cap_on_current_day * factor_data['SIZE']).sum() / market_cap_on_current_day.sum()

size_exposure = winsorization_and_market_cap_weighed_standardization(np.log(market_cap_on_current_day), market_cap_on_current_day)

#merged_size_exposure = pd.concat([size_exposure + 0.41969517155519914, pd.Series(factor_data['SIZE'].values, index = factor_data['order_book_id'].values.tolist())], axis = 1)

barra_original_size_exposure = pd.Series(factor_data['SIZE'].values, index = factor_data['order_book_id'].values.tolist())

merged_size_exposure = pd.concat([size_exposure, pd.Series(factor_data['SIZE'].values, index = factor_data['order_book_id'].values.tolist())], axis = 1)

print('min', merged_size_exposure.min())

print('mean', merged_size_exposure.mean())

print('max', merged_size_exposure.max())

print('corr', merged_size_exposure.corr())

print('tail', merged_size_exposure.tail())


### 贝塔因子比对

factor_data = pd.read_csv('/Users/jjj728/git/cne5_factors/factor_exposure/data/CNE5S_LOCALID_180427.RSK.csv')


market_cap_on_current_day = pd.Series(data = factor_data['CAPITALIZATION'].values, index = factor_data['order_book_id'].values.tolist())

original_beta_exposure = pd.Series(data = factor_data['BETA'].values, index = factor_data['order_book_id'].values.tolist())

historical_beta = pd.Series(data = factor_data['HBETA'].values, index = factor_data['order_book_id'].values.tolist())


original_beta, computed_beta_exposure = get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day)


merged_beta_exposure = pd.concat([computed_beta_exposure, original_beta_exposure], axis = 1)

merged_beta = pd.concat([original_beta, historical_beta], axis = 1)


print('min', original_beta.min(), historical_beta.min())

print('mean',  original_beta.mean(), historical_beta.mean())

print('max',  original_beta.max(), historical_beta.max())

print('corr', merged_beta.corr())

print('tail', merged_beta.tail())


print('min', computed_beta_exposure.min(), original_beta_exposure.min())

print('mean',  computed_beta_exposure.mean(), original_beta_exposure.mean())

print('max',  computed_beta_exposure.max(), original_beta_exposure.max())

print('corr', merged_size_exposure.corr())

print('tail', merged_size_exposure.tail())



date = '2018-02-06'

latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

original_beta, computed_beta_exposure = get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day)

beta_exposure = rqdatac.barra.get_factor_exposure(stock_list, latest_trading_date, latest_trading_date, 'CNE5S_BETA')

beta_exposure.index = beta_exposure.index.droplevel('date')

merged_beta_exposure = pd.concat([computed_beta_exposure, beta_exposure], axis = 1)


print('min', computed_beta_exposure.min(), beta_exposure.min())

print('mean',  computed_beta_exposure.mean(), beta_exposure.mean())

print('max',  computed_beta_exposure.max(), beta_exposure.max())

print('corr', merged_size_exposure.corr())

print('tail', merged_size_exposure.tail())
