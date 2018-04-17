

###### 市场中性 RQ 风格因子收益计算 ######

### 模块说明 ###

# 本模块分为两个部分：（1）按照市值的大、中、小，和基准组合贝塔值的高、中、低，把经筛选后的全市场股票分为九类，并进行标记；

#（2）计算市场中性 RQ 风格因子收益。这里的市场中性，定义为贝塔中性、市值中性、且多空两部分的大、中、小市值股票配比相当。


###### 部分函数说明 ######

# market_cap_and_benchmark_beta_stock_classifcation：按股票的贝塔值和市值，对其所属类别进行标记；

# market_neutralized_stock_weight：计算市场中性因子组合中各个个股的权重。目前的实现中，所有股票权重相同，保证市场中性因子组合中个股市值相等；

# factor_return_rate：计算市场中性因子收益；

# neutralization_examination：计算市场中性因子组合的贝塔值，检查市场中性化处理的效果。


###### 输出数据 ######

### 市场中性 RQ 风格因子收益；


###### 工程备忘 ######

### 1 本模块使用经过筛选后的股票池进行因子收益计算，具体的筛选标准可参看 estimation_universe 模块；

### 2 本模块使用的收益率均为算数收益率，保证个股等市值的条件下，投资组合的算术收益率等于其包含的个股收益率的算术收益率的平均值；

### 3 在计算市场中性化因子过程中，考虑了实际交易的情况，当前交易日持有的市场中性因子组合，其收益在下一个交易日计算；

### 4 对于贝塔值因子和规模因子，其市场化中性因子组合仍然留有一定的风险敞口：

# (1) 贝塔值所对应的市场化中性因子组合的贝塔值约在 0.12 ~ 0.13 之间，其它因子所对应的市场化中性因子组合的贝塔值均小于 0.01 ;

# (2) 规模因子中，多头部分的大市值股票明显多于小市值股票；而空头部分的小市值股票明显多于多头部分。

### 5 在 RQBeta 的下一阶段，纯因子收益也会在本模块计算。


### import external packages ###

import numpy as np
import pandas as pd

import datetime
import pickle
import os.path


### paths for saving files ###

temp_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/temp/"

results_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/results/"




def market_cap_and_benchmark_beta_stock_classifcation(market_cap_on_current_day, benchmark_beta_on_current_day):
     
  
    # Split the stock universe into three groups based on stocks' market cap.
      
    market_cap_on_current_day = market_cap_on_current_day.sort_values(ascending = False).dropna()
    
    large_market_cap_stock_list = market_cap_on_current_day.index[:int(len(market_cap_on_current_day)/3)]

    medium_market_cap_stock_list = market_cap_on_current_day.index[int(len(market_cap_on_current_day)/3):int(len(market_cap_on_current_day)*(2/3))]

    small_market_cap_stock_list = market_cap_on_current_day.index[int(len(market_cap_on_current_day)*(2/3)):]

     
    
    # Split the stock universe into three groups based on stocks' benchmark beta.
 
    benchmark_beta_on_current_day = benchmark_beta_on_current_day.sort_values(ascending = False).dropna()
    
    high_benchmark_beta_stock_list = benchmark_beta_on_current_day.index[:int(len(benchmark_beta_on_current_day)/3)]

    medium_benchmark_beta_stock_list = benchmark_beta_on_current_day.index[int(len(benchmark_beta_on_current_day)/3):int(len(benchmark_beta_on_current_day)*(2/3))]

    low_benchmark_beta_stock_list = benchmark_beta_on_current_day.index[int(len(benchmark_beta_on_current_day)*(2/3)):]
    
    
    # Subset the 9 different groups of the stock universe.
    
    small_and_low_list = [stock for stock in small_market_cap_stock_list if stock in low_benchmark_beta_stock_list]
            
        
    small_and_medium_list = [stock for stock in small_market_cap_stock_list if stock in medium_benchmark_beta_stock_list]
    
    
    small_and_high_list = [stock for stock in small_market_cap_stock_list if stock in high_benchmark_beta_stock_list]
    

    medium_and_low_list = [stock for stock in medium_market_cap_stock_list if stock in low_benchmark_beta_stock_list]
            
        
    medium_and_medium_list = [stock for stock in medium_market_cap_stock_list if stock in medium_benchmark_beta_stock_list]
    
    
    medium_and_high_list = [stock for stock in medium_market_cap_stock_list if stock in high_benchmark_beta_stock_list]
                              
    
    large_and_low_list = [stock for stock in large_market_cap_stock_list if stock in low_benchmark_beta_stock_list]
            
        
    large_and_medium_list = [stock for stock in large_market_cap_stock_list if stock in medium_benchmark_beta_stock_list]
    

    large_and_high_list = [stock for stock in large_market_cap_stock_list if stock in high_benchmark_beta_stock_list]
    
    
    market_neturalization_stock_list= [small_and_low_list, small_and_medium_list, small_and_high_list,\
                                       medium_and_low_list, medium_and_medium_list, medium_and_high_list,\
                                       large_and_low_list, large_and_medium_list, large_and_high_list]
                     
    return market_neturalization_stock_list



def market_neutralized_stock_weight(market_neturalization_stock_list, factor_exposure_on_current_day):
        
       
    # Descending order of the factor exposure 
                
    small_low_index = factor_exposure_on_current_day[market_neturalization_stock_list[0]].dropna().sort_values(ascending=False).index

    small_medium_index = factor_exposure_on_current_day[market_neturalization_stock_list[1]].dropna().sort_values(ascending=False).index

    small_high_index = factor_exposure_on_current_day[market_neturalization_stock_list[2]].dropna().sort_values(ascending=False).index

    medium_low_index = factor_exposure_on_current_day[market_neturalization_stock_list[3]].dropna().sort_values(ascending=False).index

    medium_medium_index = factor_exposure_on_current_day[market_neturalization_stock_list[4]].dropna().sort_values(ascending=False).index

    medium_high_index = factor_exposure_on_current_day[market_neturalization_stock_list[5]].dropna().sort_values(ascending=False).index

    large_low_index = factor_exposure_on_current_day[market_neturalization_stock_list[6]].dropna().sort_values(ascending=False).index

    large_medium_index = factor_exposure_on_current_day[market_neturalization_stock_list[7]].dropna().sort_values(ascending=False).index

    large_high_index = factor_exposure_on_current_day[market_neturalization_stock_list[8]].dropna().sort_values(ascending=False).index
     
  
  
    # long the stocks with top 1/2 factor exposures, short the stocks with small factor exposure
  
    long_portion_stock_list =  list(small_low_index[:int(len(small_low_index)/2)]) + list(small_medium_index[:int(len(small_medium_index)/2)]) \
                             + list(small_high_index[:int(len(small_high_index)/2)]) + list(medium_low_index[:int(len(medium_low_index)/2)]) \
                             + list(medium_medium_index[:int(len(medium_medium_index)/2)]) + list(medium_high_index[:int(len(medium_high_index)/2)]) \
                             + list(large_low_index[:int(len(large_low_index)/2)]) + list(large_medium_index[:int(len(large_medium_index)/2)]) \
                             + list(large_high_index[:int(len(large_high_index)/2)]) 
                            
    short_portion_stock_list =  list(small_low_index[int(len(small_low_index)/2):]) + list(small_medium_index[int(len(small_medium_index)/2):]) \
                              + list(small_high_index[int(len(small_high_index)/2):]) + list(medium_low_index[int(len(medium_low_index)/2):]) \
                              + list(medium_medium_index[int(len(medium_medium_index)/2):]) + list(medium_high_index[int(len(medium_high_index)/2):]) \
                              + list(large_low_index[int(len(large_low_index)/2):]) + list(large_medium_index[int(len(large_medium_index)/2):]) \
                              + list(large_high_index[int(len(large_high_index)/2):]) 



    # Stocks in long- or short portions are equally weighted 

    market_neutralized_long_stock_weight = (1/len(long_portion_stock_list))*np.ones(len(long_portion_stock_list))

    market_neutralized_long_stock_weight = pd.Series(market_neutralized_long_stock_weight, index = long_portion_stock_list)
  
    market_neutralized_short_stock_weight = - (1/len(short_portion_stock_list))*np.ones(len(short_portion_stock_list))

    market_neutralized_short_stock_weight = pd.Series(market_neutralized_short_stock_weight, index = short_portion_stock_list)
     
    return market_neutralized_long_stock_weight, market_neutralized_short_stock_weight






######  function for calculating the market-neutralized factor return rate #####


def factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day):
   
    ### market neutralized factor portfolio daily excess return rate ### 
  
    market_neutralized_long_portion_return_rate = np.dot(market_neutralized_long_stock_weight.values, stock_daily_excess_return_rate_on_current_day[market_neutralized_long_stock_weight.index].values)
    
    market_neutralized_short_portion_return_rate = np.dot(market_neutralized_short_stock_weight.values, stock_daily_excess_return_rate_on_current_day[market_neutralized_short_stock_weight.index].values)

    market_neutralized_factor_return_rate = market_neutralized_long_portion_return_rate + market_neutralized_short_portion_return_rate
    
    return market_neutralized_factor_return_rate



######  function for benchmark beta neutralization examination #####

def neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight,\
                               benchmark_beta_on_current_day):


    # long-portion benchmark beta

    market_neutralized_long_portion_benchmark_beta = np.dot(market_neutralized_long_stock_weight.values, benchmark_beta_on_current_day[market_neutralized_long_stock_weight.index].values)

    # short-portion benchmark beta

    market_neutralized_short_portion_benchmark_beta = np.dot(market_neutralized_short_stock_weight.values, benchmark_beta_on_current_day[market_neutralized_short_stock_weight.index].values)

    # benchmark beta for the 

    market_neutralized_benchmark_beta = market_neutralized_long_portion_benchmark_beta + market_neutralized_short_portion_benchmark_beta
    
    return market_neutralized_benchmark_beta
           


def market_neutrlaized_factor_return():
    
    
    ### load listed stocks ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks.pkl")
    
    pkfl = open(complete_path,'rb')

    listed_stocks = pickle.load(pkfl)

    pkfl.close()

    
    ### load estimation univese ###

    complete_path = os.path.join(temp_path, "estimation_universe.pkl")
    
    pkfl = open(complete_path,'rb')

    estimation_universe = pickle.load(pkfl)

    pkfl.close()


    ### load stock daily excess return rate ###
    
    complete_path = os.path.join(temp_path, "stock_daily_excess_return_rate.pkl")
    
    pkfl = open(complete_path,'rb')

    stock_daily_excess_return_rate = pickle.load(pkfl)

    pkfl.close()


    ### load benchmark beta ###
    
    complete_path = os.path.join(temp_path, "benchmark_beta.pkl")
    
    pkfl = open(complete_path,'rb')

    benchmark_beta = pickle.load(pkfl)

    pkfl.close()


    ### load fundamental data ###
    
    complete_path = os.path.join(temp_path, "df_fundamental.pkl")
    
    pkfl = open(complete_path,'rb')

    df_fundamental = pickle.load(pkfl)

    pkfl.close()
    
  
    ### market cap ###
    
    market_cap = df_fundamental['market_cap']


    ### std_benchmark_beta_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_benchmark_beta_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_benchmark_beta_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_momentum_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_momentum_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_momentum_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_reversal_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_reversal_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_reversal_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_size_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_size_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_size_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_earning_yield_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_earning_yield_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_earning_yield_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_volatility_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_volatility_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_volatility_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_growth_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_growth_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_growth_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_value_missing_data_imputed ###

    complete_path = os.path.join(results_path, "std_value_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_value_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_leverage_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_leverage_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_leverage_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### std_liquidity_missing_data_imputed ###

    complete_path = os.path.join(temp_path, "std_liquidity_missing_data_imputed.pkl")
    
    pkfl = open(complete_path,'rb')

    std_liquidity_missing_data_imputed = pickle.load(pkfl)

    pkfl.close()


    ### market-neutralized factor return ###

    factors_name = ['benchmark_beta', 'momentum', 'reversal', 'size', 'earning_yield','volatility', 'growth', 'value', 'leverage', 'liquidity'] 

    estimation_universe_t = estimation_universe.transpose()

    # remove the "%H-%M-%S" in the time-index

    stock_daily_excess_return_rate.index = [datetime.datetime.strptime(i.strftime("%Y-%m-%d"),"%Y-%m-%d").date() for i in stock_daily_excess_return_rate.index]


    # skip the first 17 trading days, the rest are the original time period from the input.

    for date in listed_stocks.index[17:] :
    
        #print(date)
    
        one_day_before = stock_daily_excess_return_rate.index.get_loc(date) - 1
    
        one_day_before = stock_daily_excess_return_rate.index[one_day_before]

    
        market_neutralized_factor_return_rate_list = [ ]
    
        # Obtain the stock list for estimation.
    
        estimation_stock_list = estimation_universe_t[estimation_universe_t[date] == 'True'].index


        # Obtain the stock list with no missing data

        # Note that the time stamp of the stock_daily_excess_return_rate is one day before

        benchmark_beta_on_current_day = benchmark_beta[estimation_stock_list].ix[one_day_before].dropna()

        market_cap_on_current_day = market_cap[estimation_stock_list].ix[one_day_before].dropna()
    
        # Note that the time stamp of the stock_daily_excess_return_rate is today.

        stock_daily_excess_return_rate_on_current_day = stock_daily_excess_return_rate[estimation_stock_list].ix[date].dropna()
    
    
        combined_lists_with_no_missing_data = [benchmark_beta_on_current_day.index.tolist(), market_cap_on_current_day.index.tolist(), stock_daily_excess_return_rate_on_current_day.index.tolist()]
    
        estimation_stock_list_no_missing_data = set.intersection(*map(set, combined_lists_with_no_missing_data))

        estimation_stock_list_no_missing_data = [x for x in estimation_stock_list_no_missing_data]


        # filter out the stock with missing data

        benchmark_beta_on_current_day = benchmark_beta_on_current_day[estimation_stock_list_no_missing_data]

        market_cap_on_current_day = market_cap_on_current_day[estimation_stock_list_no_missing_data]

        stock_daily_excess_return_rate_on_current_day = stock_daily_excess_return_rate_on_current_day[estimation_stock_list_no_missing_data]
        
        
        # Obtain the 9 groups of stocks with differnt market cap and benchmark beta

        market_neturalization_stock_list\
        = market_cap_and_benchmark_beta_stock_classifcation(market_cap_on_current_day, benchmark_beta_on_current_day)

    
    
        ######  BENCHMARK BETA FACTOR ######
    
        benchmark_beta_factor_exposure_on_current_day = std_benchmark_beta_missing_data_imputed.ix[one_day_before]


        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, benchmark_beta_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_benchmark_beta_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_benchmark_beta_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)

     
        ######  MOMENTUM FACTOR ######
    
        momentum_factor_exposure_on_current_day = std_momentum_missing_data_imputed.ix[one_day_before]

    
        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, momentum_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_momentum_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_momentum_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)

      
        ###### REVERSAL FACTOR ######
    
        reversal_factor_exposure_on_current_day = std_reversal_missing_data_imputed.ix[one_day_before]

    
        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, reversal_factor_exposure_on_current_day)

        # Calculate the market neutralized factor return rate

        market_neutralized_reversal_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_reversal_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)


        ###### SIZE FACTOR ######
    
        size_factor_exposure_on_current_day = std_size_missing_data_imputed.ix[one_day_before]

    
        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, size_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_size_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_size_factor_return_rate)
        
        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)



        ###### EARNING YIELD FACTOR ######
    
        earning_yield_factor_exposure_on_current_day = std_earning_yield_missing_data_imputed.ix[one_day_before]


        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, earning_yield_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_earning_yield_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_earning_yield_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)



        ###### VOLATILITY FACTOR ######
    
        volatility_factor_exposure_on_current_day = std_volatility_missing_data_imputed.ix[one_day_before]


        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, volatility_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_volatility_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_volatility_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)


        ###### GROWTH FACTOR ######
    
        growth_factor_exposure_on_current_day = std_growth_missing_data_imputed.ix[one_day_before]


        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, growth_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_growth_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_growth_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)



        ###### VALUE FACTOR ######
    
        value_factor_exposure_on_current_day = std_value_missing_data_imputed.ix[one_day_before]

    
        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, value_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_value_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_value_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)



        ###### LEVERAGE FACTOR ######
    
        leverage_factor_exposure_on_current_day = std_leverage_missing_data_imputed.ix[one_day_before]


        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, leverage_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_leverage_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_leverage_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)

    
    
        ###### LIQUIDITY FACTOR ######
    
        liquidity_factor_exposure_on_current_day = std_liquidity_missing_data_imputed.ix[one_day_before]


        # Obtain the market neutralized stock weight for long- and short- portions
    
        market_neutralized_long_stock_weight, market_neutralized_short_stock_weight \
        = market_neutralized_stock_weight(market_neturalization_stock_list, liquidity_factor_exposure_on_current_day)


        # Calculate the market neutralized factor return rate

        market_neutralized_liquidity_factor_return_rate \
        = factor_return_rate(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, stock_daily_excess_return_rate_on_current_day)

        market_neutralized_factor_return_rate_list.append(market_neutralized_liquidity_factor_return_rate)


        # benchmark beta neutralization examination

        market_neutralized_benchmark_beta = neutralization_examination(market_neutralized_long_stock_weight, market_neutralized_short_stock_weight, benchmark_beta_on_current_day)



        if date == listed_stocks.index[17] :
        
            # Create the dataframes
        
            df_market_neutralized_factor_return_rate = pd.DataFrame(market_neutralized_factor_return_rate_list, index = factors_name, columns = [date])
       
        else :

            series_market_neutralized_factor_return_rate = pd.DataFrame(market_neutralized_factor_return_rate_list, index = factors_name, columns = [date])
       
   
            df_market_neutralized_factor_return_rate = pd.concat([df_market_neutralized_factor_return_rate, series_market_neutralized_factor_return_rate], axis=1, join='inner')
       
 

    df_market_neutralized_factor_return_rate = df_market_neutralized_factor_return_rate.transpose()


    ### output market_neutralized_factor_return_rate_list ###

    complete_path = os.path.join(results_path, "market_neutralized_factor_return.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_market_neutralized_factor_return_rate, output)

    output.close()
    
    # print a message 
    
    print('market-neutralized factor return estimation is done')




def factor_return_estimation():

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    
    print('factor return estimation begins')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    market_neutrlaized_factor_return()
    
    # pure factor return estimation will be implemented here in the future.
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    
    print('factor return estimation is done')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    
    

    complete_path = os.path.join(results_path, "market_neutralized_factor_return.pkl")
    
    pkfl = open(complete_path,'rb')

    market_neutralized_factor_return = pickle.load(pkfl)

    pkfl.close()


