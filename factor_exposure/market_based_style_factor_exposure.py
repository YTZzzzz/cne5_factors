

###### 细分因子暴露度及权重计算 ######


### 模块说明 ###

# 本模块分为三部分：（1）计算不同半衰期的指数权重序列；（2）计算细分因子，及部分风格因子的暴露度；（3）用随机森林回归分析确定细分因子权重。

# 本模块计算以下风格因子的暴露度：（1）贝塔值因子；（2）反转因子。

# 本模块计算以下风格因子的细分因子暴露度，及其权重：（1）动量因子；（2）波动率因子；（3）盈利率因子；（4）成长性因子；（5）流动性因子。


### 部分函数说明 ###

# winsorization_and_standardization : 对细分因子进行标准化和Winsorization处理，保证随机森林回归分析中，各细分因子具有大致相同的影响力。


###### 输出数据 ######

#（1）细分因子，及部分风格因子的暴露度；

#（2）细分因子权重。


###### 工程备忘 ######

### 1 在计算各个量价因子（贝塔值、动量、反转、波动率和流动性）时，我们按照以下方式处理缺失值：

# （1）在计算个股的贝塔值因子暴露度时，在 200 个交易日中，如果缺失值少于 66 个（约有 6 个月数据），则剔除缺失值计算贝塔值暴露度；否则定为缺失值；

# （2）在计算其它量价因子的细分因子时，如果时间序列中的缺失值多于一半，则定为缺失值；如果不多于一半，则使用最近交易日的历史数据，对未来交易日缺失值进行填补（通过 Pandas 的 fillna(method = 'pad') 实现。


#### 2 在计算短期波动率（残余风险）时，我们需要前 21 个交易日的贝塔因子暴露度数据，因此贝塔值暴露度计算额外增加 21 个交易日的计算。


#### 3 流动性因子所使用的日均换手率数据，是直接通过交易量和流通股本算得，而非从 RQData 相关 API 读取。



### import external packages ###

import numpy as np
import pandas as pd
import sys

sys.path.append("/Users/jjj728/git/barra_models_implementation/barra_cne5_factors/factor_exposure")

from intermediate_variables import *
from operators import *


def short_term_volatility_exposure(listed_stocks, stock_daily_excess_return, benchmark_daily_excess_return, date):


    exp_weight = get_exponential_weight(half_life = 42, length = 252)

    start_date = rqdatac.get_trading_dates()[-252:]

    get_daily_excess_return_rate(stock_list = listed_stocks, benchmark_id, start_date, end_date)


    benchmark_beta = get_benchmark_beta()

    # Obtain the list of listed stock at current trading day.

    stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()

    # Obtain the begindate and enddate of the variables.

    stock_today = stock_daily_excess_return_rate.index.get_loc(date)

    stock_today = stock_daily_excess_return_rate.index[stock_today]

    stock_one_month_before = stock_daily_excess_return_rate.index.get_loc(date) - 21

    stock_one_month_before = stock_daily_excess_return_rate.index[stock_one_month_before]

    benchmark_beta_today = benchmark_beta.index.get_loc(date)

    benchmark_beta_today = benchmark_beta.index[benchmark_beta_today]

    benchmark_beta_one_month_before = benchmark_beta.index.get_loc(date) - 21

    benchmark_beta_one_month_before = benchmark_beta.index[benchmark_beta_one_month_before]

    benchmark_return_today = benchmark_daily_excess_return_rate.index.get_loc(date)

    benchmark_return_today = benchmark_daily_excess_return_rate.index[benchmark_return_today]

    benchmark_return_one_month_before = benchmark_daily_excess_return_rate.index.get_loc(date) - 21

    benchmark_return_one_month_before = benchmark_daily_excess_return_rate.index[benchmark_return_one_month_before]

    # Subset the dataframe.

    # benchmark_component = benchmark_beta * benchmark_daily_excess_return_rate

    benchmark_component = pd.DataFrame(
        benchmark_beta[stock_list].ix[benchmark_beta_one_month_before:benchmark_beta_today].transpose().values * \
        benchmark_daily_excess_return_rate.ix[benchmark_return_one_month_before:benchmark_return_today].values, \
        index=benchmark_beta[stock_list].ix[benchmark_beta_one_month_before:benchmark_beta_today].columns, \
        columns=benchmark_beta[stock_list].ix[benchmark_beta_one_month_before:benchmark_beta_today].index)

    benchmark_component = benchmark_component.transpose()

    # residual_return = excess_return - benchmark_component

    # if excess return or benchmark_component is NAN, then the short_term_volatility_series is NAN.

    residual_return = stock_daily_excess_return_rate[stock_list].ix[stock_one_month_before:stock_today].subtract(
        benchmark_component)

    ### remove the mean from the time series of stock_daily_excess_return_rate

    mean_of_residual_return = residual_return.mean(axis=0)

    rescaled_residual_return = residual_return.sub(mean_of_residual_return, axis=1)

    ### Take the square

    squared_short_term_volatility = rescaled_residual_return.apply(np.square)

    ### Obtain the stock list with more than 22/2 = 11 NANs ###

    nan_count = squared_short_term_volatility.isnull().sum()

    nan_stock_list = nan_count[nan_count > 11].index

    # Use the data from past trading days to replace the NANs.

    squared_short_term_volatility = squared_short_term_volatility.fillna(method='pad')

    ### np.nan_to_num replace NANs with ZEROs, otherwise the results is NAN if squared_short_term_volatility contains NAN.

    weighted_short_term_volatility = np.dot(np.nan_to_num(squared_short_term_volatility.transpose()),
                                            exp_weight_11.values[0:len(squared_short_term_volatility.index)])

    sqrt_weighted_short_term_volatility = np.sqrt(weighted_short_term_volatility)

    # put the results into a dataframe

    if date == listed_stocks.index[0]:

        df_sqrt_weighted_short_term_volatility = pd.DataFrame(sqrt_weighted_short_term_volatility,
                                                              index=residual_return.columns, columns=[date])

        # Assign NAN to the stocks with more than 44 NANs.

        df_sqrt_weighted_short_term_volatility.ix[nan_stock_list] = np.nan

    else:

        series_sqrt_weighted_short_term_volatility = pd.DataFrame(sqrt_weighted_short_term_volatility,
                                                                  index=residual_return.columns, columns=[date])

        # Assign NAN to the stocks with more than 44 NANs.

        series_sqrt_weighted_short_term_volatility.ix[nan_stock_list] = np.nan

        # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.

        df_sqrt_weighted_short_term_volatility = pd.concat(
            [df_sqrt_weighted_short_term_volatility, series_sqrt_weighted_short_term_volatility], axis=1, join='outer')


    df_sqrt_weighted_short_term_volatility = df_sqrt_weighted_short_term_volatility.transpose()

    return df_sqrt_weighted_short_term_volatility







def three_month_momentum_exposure(exp_weight_22, listed_stocks, stock_daily_excess_return_rate):

    for date in listed_stocks.index :
        
        # Obtain the list of listed stock at current trading day.
 
        stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()
    
        # Obtain the begin date and end date of the three month momentum. Skip the recent 22 trading days to avoid reversal effect.

        one_month_before = stock_daily_excess_return_rate.index.get_loc(date) - 22

        one_month_before = stock_daily_excess_return_rate.index[one_month_before]
        
        
        three_month_before = stock_daily_excess_return_rate.index.get_loc(date) - 65

        three_month_before = stock_daily_excess_return_rate.index[three_month_before]
        
  
        three_month_momentum = stock_daily_excess_return_rate[stock_list].ix[three_month_before:one_month_before].copy()
    
        log_three_month_momentum = np.log(1 + three_month_momentum)
    
        ### Obtain the stock list with more than 44 - 22 = 22 NANs, that means at least data of 22 trading days (1 months) is available ###

        nan_count = log_three_month_momentum.isnull().sum() 
    
        nan_stock_list = nan_count[nan_count > 22].index
        
        # Use the data from past trading days to replace the NANs.
        
        log_three_month_momentum = log_three_month_momentum.fillna(method='pad')
    
    
        # the dot product function of NumPy has triky behaviors dealing with NANs, check out the link below.
        # http://stackoverflow.com/questions/23374524/numpy-dot-bug-inconsistent-nan-behavior

        ### np.nan_to_num replace NANs with ZEROs, otherwise the results is NAN if weighted_log_three_month_momentum contains NAN.
        
        weighted_log_three_month_momentum = np.dot(np.nan_to_num(log_three_month_momentum.values.transpose()), exp_weight_22.values[0:len(three_month_momentum.index)])
    
    
        # put the results into a dataframe
    
        if date == listed_stocks.index[0] :
                
            df_weighted_log_three_month_momentum = pd.DataFrame(weighted_log_three_month_momentum, index = log_three_month_momentum.columns, columns = [date])
              
            # Assign NAN to the stocks with more than 22 NANs.
        
            df_weighted_log_three_month_momentum.ix[nan_stock_list] = np.nan

        else :
                
            series_weighted_log_three_month_momentum = pd.DataFrame(weighted_log_three_month_momentum, index = log_three_month_momentum.columns, columns = [date])
    
            # Assign NAN to the stocks with more than 22 NANs.
        
            series_weighted_log_three_month_momentum.ix[nan_stock_list] = np.nan
            
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_weighted_log_three_month_momentum = pd.concat([df_weighted_log_three_month_momentum, series_weighted_log_three_month_momentum], axis=1, join='outer')
    
    
    df_weighted_log_three_month_momentum = df_weighted_log_three_month_momentum.transpose()

    return df_weighted_log_three_month_momentum
    
    



def six_month_momentum_exposure(exp_weight_55, listed_stocks, stock_daily_excess_return_rate):


    for date in listed_stocks.index :
        
        # Obtain the order_book_id list of listed stock at current trading day.

        stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()
    
    
        # Obtain the begin date and end date of the six month momentum. Skip the recent 22 trading days to avoid reversal effect.

        one_month_before = stock_daily_excess_return_rate.index.get_loc(date) - 22
        
        one_month_before = stock_daily_excess_return_rate.index[one_month_before]

        
        six_month_before = stock_daily_excess_return_rate.index.get_loc(date) - 131
        
        six_month_before = stock_daily_excess_return_rate.index[six_month_before]
       
  
        six_month_momentum = stock_daily_excess_return_rate[stock_list].ix[six_month_before:one_month_before].copy()
    
        log_six_month_momentum = np.log(1 + six_month_momentum)
    
        ### Obtain the stock list with more than 110/2 = 55 NANs ###

        nan_count = log_six_month_momentum.isnull().sum() 
    
        nan_stock_list = nan_count[nan_count > 55].index
        
        # Use the data from past trading days to replace the NANs.
        
        log_six_month_momentum = log_six_month_momentum.fillna(method='pad')

    
        # the dot product of NumPy has triky behaviors dealing with NAN, check out the link below.
        # http://stackoverflow.com/questions/23374524/numpy-dot-bug-inconsistent-nan-behavior

        ### np.nan_to_num replace NANs with ZEROs, otherwise the results is NAN if weighted_log_six_month_momentum contains NAN.
         
        weighted_log_six_month_momentum = np.dot(np.nan_to_num(log_six_month_momentum.values.transpose()), exp_weight_55.values[0:len(six_month_momentum.index)])
    
        # put the results into a dataframe
    
        if date == listed_stocks.index[0] :
            
            df_weighted_log_six_month_momentum = pd.DataFrame(weighted_log_six_month_momentum, index = log_six_month_momentum.columns, columns = [date])
         
            # Assign NAN to the stocks with more than 44 NANs.
        
            df_weighted_log_six_month_momentum.ix[nan_stock_list] = np.nan
 
        else :
            
            series_weighted_log_six_month_momentum = pd.DataFrame(weighted_log_six_month_momentum, index = log_six_month_momentum.columns, columns = [date])

            # Assign NAN to the stocks with more than 44 NANs.
        
            series_weighted_log_six_month_momentum.ix[nan_stock_list] = np.nan

            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_weighted_log_six_month_momentum = pd.concat([df_weighted_log_six_month_momentum, series_weighted_log_six_month_momentum], axis=1, join='outer')
    

    df_weighted_log_six_month_momentum = df_weighted_log_six_month_momentum.transpose()

    return df_weighted_log_six_month_momentum
       










def medium_term_volatility_exposure(exp_weight_22, listed_stocks, stock_daily_excess_return_rate):

    for date in listed_stocks.index :
        
        # Obtain the list of listed stock at current trading day.
        
        stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()
    
    
        # Obtain the begindate and enddate of the variables.

        stock_today = stock_daily_excess_return_rate.index.get_loc(date)

        stock_today = stock_daily_excess_return_rate.index[stock_today]
    
        stock_three_month_before = stock_daily_excess_return_rate.index.get_loc(date) - 65

        stock_three_month_before = stock_daily_excess_return_rate.index[stock_three_month_before]


        # Subset the dataframe.
  
        medium_term_volatility_series = stock_daily_excess_return_rate[stock_list].ix[stock_three_month_before:stock_today].copy()
    
    
        ### remove the mean from the time series of stock_daily_excess_return_rate
    
        mean_of_medium_term_volatility_series = medium_term_volatility_series.mean(axis = 0)

        rescaled_medium_term_volatility = medium_term_volatility_series.sub(mean_of_medium_term_volatility_series, axis = 1)


        ### Take the square 

        squared_medium_term_volatility = rescaled_medium_term_volatility.apply(np.square)


        ### Obtain the stock list with more than 66/2 = 33 NANs ###

        nan_count = squared_medium_term_volatility.isnull().sum() 
    
        nan_stock_list = nan_count[nan_count > 33].index
    
        # Use the data from past trading days to replace the NANs.
        
        squared_medium_term_volatility = squared_medium_term_volatility.fillna(method='pad')
   
        ### np.nan_to_num replace NANs with ZEROs, otherwise the results is NAN if squared_medium_term_volatility contains NAN.
    
        weighted_medium_term_volatility = np.dot(np.nan_to_num(squared_medium_term_volatility.transpose()), exp_weight_22.values[0:len(squared_medium_term_volatility.index)])
    
        sqrt_weighted_medium_term_volatility = np.sqrt(weighted_medium_term_volatility)
    
        # put the results into a dataframe
    
        if date == listed_stocks.index[0] :
            
            df_sqrt_weighted_medium_term_volatility = pd.DataFrame(sqrt_weighted_medium_term_volatility, index = medium_term_volatility_series.columns, columns = [date])
        
            # Assign NAN to the stocks with more than 33 NANs.
        
            df_sqrt_weighted_medium_term_volatility.ix[nan_stock_list] = np.nan
        
        else :
                
            series_sqrt_weighted_medium_term_volatility = pd.DataFrame(sqrt_weighted_medium_term_volatility, index = medium_term_volatility_series.columns, columns = [date])

            # Assign NAN to the stocks with more than 44 NANs.
                
            series_sqrt_weighted_medium_term_volatility.ix[nan_stock_list] = np.nan
        
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_sqrt_weighted_medium_term_volatility = pd.concat([df_sqrt_weighted_medium_term_volatility, series_sqrt_weighted_medium_term_volatility], axis=1, join='outer')


    df_sqrt_weighted_medium_term_volatility = df_sqrt_weighted_medium_term_volatility.transpose()

    return df_sqrt_weighted_medium_term_volatility






def long_term_volatility_exposure(listed_stocks, stock_daily_excess_return_rate):


    for date in listed_stocks.index :
        
        # Obtain the order_book_id list of listed stock at current trading day.
        
        stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()
      
      
        # Obtain the begindate and enddate of the long-term volatility.

        stock_today = stock_daily_excess_return_rate.index.get_loc(date)
        
        stock_today = stock_daily_excess_return_rate.index[stock_today]
        
        end_of_first_period = stock_daily_excess_return_rate.index.get_loc(date) - 21

        end_of_first_period = stock_daily_excess_return_rate.index[end_of_first_period]
        

        begin_of_second_period = stock_daily_excess_return_rate.index.get_loc(date) - 22

        begin_of_second_period = stock_daily_excess_return_rate.index[begin_of_second_period]

        end_of_second_period = stock_daily_excess_return_rate.index.get_loc(date) - 43
        
        end_of_second_period = stock_daily_excess_return_rate.index[end_of_second_period]


        begin_of_third_period = stock_daily_excess_return_rate.index.get_loc(date) - 44

        begin_of_third_period = stock_daily_excess_return_rate.index[begin_of_third_period]

        end_of_third_period = stock_daily_excess_return_rate.index.get_loc(date) - 65
        
        end_of_third_period = stock_daily_excess_return_rate.index[end_of_third_period]


        begin_of_forth_period = stock_daily_excess_return_rate.index.get_loc(date) - 66

        begin_of_forth_period = stock_daily_excess_return_rate.index[begin_of_forth_period]

        end_of_forth_period = stock_daily_excess_return_rate.index.get_loc(date) - 87
        
        end_of_forth_period = stock_daily_excess_return_rate.index[end_of_forth_period]

 
        begin_of_fifth_period = stock_daily_excess_return_rate.index.get_loc(date) - 88

        begin_of_fifth_period = stock_daily_excess_return_rate.index[begin_of_fifth_period]
   
        end_of_fifth_period = stock_daily_excess_return_rate.index.get_loc(date) - 109
        
        end_of_fifth_period = stock_daily_excess_return_rate.index[end_of_fifth_period]


        begin_of_sixth_period = stock_daily_excess_return_rate.index.get_loc(date) - 110

        begin_of_sixth_period = stock_daily_excess_return_rate.index[begin_of_sixth_period]
    
        end_of_sixth_period = stock_daily_excess_return_rate.index.get_loc(date) - 131
        
        end_of_sixth_period = stock_daily_excess_return_rate.index[end_of_sixth_period]


        # subset the dataframe.
  
        first_period_return_rate = stock_daily_excess_return_rate[stock_list].ix[end_of_first_period:stock_today].copy()
    
        second_period_return_rate = stock_daily_excess_return_rate[stock_list].ix[end_of_second_period:begin_of_second_period].copy()

        third_period_return_rate = stock_daily_excess_return_rate[stock_list].ix[end_of_third_period:begin_of_third_period].copy()

        forth_period_return_rate = stock_daily_excess_return_rate[stock_list].ix[end_of_forth_period:begin_of_forth_period].copy()

        fifth_period_return_rate = stock_daily_excess_return_rate[stock_list].ix[end_of_fifth_period:begin_of_fifth_period].copy()

        sixth_period_return_rate = stock_daily_excess_return_rate[stock_list].ix[end_of_sixth_period:begin_of_sixth_period].copy()
    
        # Take the logarithm
    
        log_first_period_return_rate = np.log(1 + first_period_return_rate)

        log_second_period_return_rate = np.log(1 + second_period_return_rate)

        log_third_period_return_rate = np.log(1 + third_period_return_rate)

        log_forth_period_return_rate = np.log(1 + forth_period_return_rate)

        log_fifth_period_return_rate = np.log(1 + fifth_period_return_rate)

        log_sixth_period_return_rate = np.log(1 + sixth_period_return_rate)
        
    
        # Sum over columns, if the entire column is filled with NANs, then the sum is NAN.
    
        sum_log_first_period_return_rate = log_first_period_return_rate.sum(axis = 0)
    
        sum_log_second_period_return_rate = log_second_period_return_rate.sum(axis = 0)

        sum_log_third_period_return_rate = log_third_period_return_rate.sum(axis = 0)

        sum_log_forth_period_return_rate = log_forth_period_return_rate.sum(axis = 0)

        sum_log_fifth_period_return_rate = log_fifth_period_return_rate.sum(axis = 0)

        sum_log_sixth_period_return_rate = log_sixth_period_return_rate.sum(axis = 0)

        # Merge the sum into one dataframe

        # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
        df_volatility_period = pd.DataFrame([sum_log_first_period_return_rate, sum_log_second_period_return_rate,\
                                             sum_log_third_period_return_rate, sum_log_forth_period_return_rate,\
                                             sum_log_fifth_period_return_rate, sum_log_sixth_period_return_rate\
                                            ], index = ['first_period', 'second_period', 'third_period',\
                                                        'forth_period', 'fifth_period', 'sixth_period', ])
                                                        
        max_volatility_period = df_volatility_period.max(axis = 0)

        min_volatility_period = df_volatility_period.min(axis = 0)
    
        # IF the all the period volatilities are NANs, then long_term_volatility is NAN.
    
        long_term_volatility = max_volatility_period - min_volatility_period
    
    
        # put the results into a dataframe
    
        if date == listed_stocks.index[0] :
           
            df_long_term_volatility = pd.DataFrame(long_term_volatility, columns = [date])
                
        else :
                
            series_long_term_volatility = pd.DataFrame(long_term_volatility, columns = [date])
        
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_long_term_volatility = pd.concat([df_long_term_volatility, series_long_term_volatility], axis=1, join='outer')


    df_long_term_volatility = df_long_term_volatility.transpose()

    return df_long_term_volatility




def liquidity_exposure(listed_stocks):

    daily_turnover_rate = get_daily_turnover_rate


    for date in listed_stocks.index :
        
        # Obtain the order_book_id list of listed stock at current trading day.

        stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()
    
    
        # Obtain the begindate and enddate of liquidities.

        today = daily_turnover_rate.index.get_loc(date)
        
        today = daily_turnover_rate.index[today]

    
        one_month_before = daily_turnover_rate.index.get_loc(date) - 21
        
        one_month_before = daily_turnover_rate.index[one_month_before]

    
        three_months_before = daily_turnover_rate.index.get_loc(date) - 65
        
        three_months_before = daily_turnover_rate.index[three_months_before]


        six_months_before = daily_turnover_rate.index.get_loc(date) - 131
        
        six_months_before = daily_turnover_rate.index[six_months_before]


        # Obtain slices of the dataframe.
  
        one_month_turnover_rate = daily_turnover_rate[stock_list].ix[one_month_before:today].copy()
    
        three_months_turnover_rate = daily_turnover_rate[stock_list].ix[three_months_before:today].copy()

        six_months_turnover_rate = daily_turnover_rate[stock_list].ix[six_months_before:today].copy()
    
    
        ### Obtain the stock list with more than 22/2 = 11 NANs ###

        one_month_turnover_rate_nan_count = one_month_turnover_rate.isnull().sum() 
    
        one_month_turnover_rate_nan_stock_list = one_month_turnover_rate_nan_count[one_month_turnover_rate_nan_count > 11].index


        ### Obtain the stock list with more than 66/2 = 33 NANs ###

        three_months_turnover_rate_nan_count = three_months_turnover_rate.isnull().sum() 
    
        three_months_turnover_rate_nan_stock_list = three_months_turnover_rate_nan_count[three_months_turnover_rate_nan_count > 33].index


        ### Obtain the stock list with more than 132/2 = 66 NANs ###

        six_months_turnover_rate_nan_count = six_months_turnover_rate.isnull().sum() 
    
        six_months_turnover_rate_nan_stock_list = six_months_turnover_rate_nan_count[six_months_turnover_rate_nan_count > 66].index

       
        # Compute the mean of columns, if the entire column is filled with NANs, then the sum is NAN.
    
        mean_one_month_turnover_rate = one_month_turnover_rate.mean(axis = 0)
    
        mean_three_months_turnover_rate = three_months_turnover_rate.mean(axis = 0)

        mean_six_months_turnover_rate = six_months_turnover_rate.mean(axis = 0)
 
    
        # put the results into a dataframe
    
        if date == listed_stocks.index[0] :
            
            df_mean_one_month_turnover_rate = pd.DataFrame(mean_one_month_turnover_rate, columns = [date])
 
            df_mean_three_months_turnover_rate = pd.DataFrame(mean_three_months_turnover_rate, columns = [date])

            df_mean_six_months_turnover_rate = pd.DataFrame(mean_six_months_turnover_rate, columns = [date])

            # Assign NANs to the stocks with unreasonable numbers of NANs.
        
            df_mean_one_month_turnover_rate.ix[one_month_turnover_rate_nan_stock_list] = np.nan

            df_mean_three_months_turnover_rate.ix[three_months_turnover_rate_nan_stock_list] = np.nan

            df_mean_six_months_turnover_rate.ix[six_months_turnover_rate_nan_stock_list] = np.nan
               
        else :
                
            series_mean_one_month_turnover_rate = pd.DataFrame(mean_one_month_turnover_rate, columns = [date])
 
            series_mean_three_months_turnover_rate = pd.DataFrame(mean_three_months_turnover_rate, columns = [date])

            series_mean_six_months_turnover_rate = pd.DataFrame(mean_six_months_turnover_rate, columns = [date])


            # Assign NAN to the stocks with unreasonable numbers of NANs.
        
            series_mean_one_month_turnover_rate.ix[one_month_turnover_rate_nan_stock_list] = np.nan

            series_mean_three_months_turnover_rate.ix[three_months_turnover_rate_nan_stock_list] = np.nan

            series_mean_six_months_turnover_rate.ix[six_months_turnover_rate_nan_stock_list] = np.nan

        
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_mean_one_month_turnover_rate = pd.concat([df_mean_one_month_turnover_rate, series_mean_one_month_turnover_rate], axis=1, join='outer')

            df_mean_three_months_turnover_rate = pd.concat([df_mean_three_months_turnover_rate, series_mean_three_months_turnover_rate], axis=1, join='outer')

            df_mean_six_months_turnover_rate = pd.concat([df_mean_six_months_turnover_rate, series_mean_six_months_turnover_rate], axis=1, join='outer')


    df_mean_one_month_turnover_rate = df_mean_one_month_turnover_rate.transpose()

    df_mean_three_months_turnover_rate = df_mean_three_months_turnover_rate.transpose()

    df_mean_six_months_turnover_rate = df_mean_six_months_turnover_rate.transpose()

    return df_mean_one_month_turnover_rate, df_mean_three_months_turnover_rate, df_mean_six_months_turnover_rate
    
    

def get_market_based_atomic_descriptors(date):


    stocks_daily_excess_return, benchmark_daily_excess_return = get_daily_excess_return_rate(stock_list = listed_stocks, benchmark_id = 'CSI300.INDX', date)

    benchmark_beta_exposure()

    three_month_momentum_exposure(exp_weight_22)

    six_month_momentum_exposure(exp_weight_55)

    reversal_exposure(exp_weight_11)

    short_term_volatility_exposure(exp_weight_11)

    medium_term_volatility_exposure(exp_weight_22)

    long_term_volatility_exposure()

    liquidity_exposure()

    atomic_descriptor_weight_estimation()

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('atomic descriptors computation is done')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  
  
