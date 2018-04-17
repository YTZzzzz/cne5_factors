
###### 风格因子暴露度计算 ######


### 模块说明 ###

# 本模块分为两部分：（1）计算风格因子的暴露度（规模、价值和杠杆）；（2）对细分因子和风格因子进行标准化、Winsorization 和细分因子填补缺失值处理。

# 除上述三个风格因子外，其余风格因子暴露度的计算均在细分因子模块（atomic_descriptors）完成。


### 部分函数说明 ###

# winsorization_and_standardization : 对细分因子进行标准化和Winsorization处理，用于风格暴露度分析和业绩归因。

# winsorization_and_market_cap_weighed_standardization : 对细分因子进行市值加权标准化和Winsorization处理，用于风险预测。

# two_atomic_descriptors_combination ：（1）对两个细分因子进行加权组合得到风格因子暴露度；（2）当个股的细分因子暴露度存在缺失值时，用其余细分因子暴露度进行替换。

# three_atomic_descriptors_combination ：（1）对三个细分因子进行加权组合得到风格因子暴露度；（2）当个股的细分因子暴露度存在缺失值时，用其余细分因子暴露度进行替换。



###### 输出数据 ######

#（1）标准化风格因子暴露度；

#（2）市值加权标准化因子暴露度（用于以后的风险预测模型）。





### import external packages ###

import numpy as np
import pandas as pd

import datetime
import pickle
import os.path

### paths for saving files ###

temp_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/temp/"

results_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/results/"



def size():
    
    ### load listed stocks ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks.pkl")
    
    pkfl = open(complete_path,'rb')

    listed_stocks = pickle.load(pkfl)

    pkfl.close()
    
    
    ### load fundamental data ###
    
    complete_path = os.path.join(temp_path, "df_fundamental.pkl")
    
    pkfl = open(complete_path,'rb')

    df_fundamental = pickle.load(pkfl)

    pkfl.close()
    
    
    ### market cap ###
    
    market_cap = df_fundamental['market_cap']


    # Take the transpose of listed_stocks to facilitate the calculation.

    listed_stocks_t = listed_stocks.transpose()


    for date in listed_stocks.index :
        
        # Obtain the order_book_id list of listed stock at current trading day.
   
        stock_list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()
      
        market_cap_at_current_date = market_cap[stock_list].ix[date].copy()
    
        # Replace all zeros with nan (potential mistakes in data !), otherwise taking the log will lead to - inf.
    
        market_cap_at_current_date.replace(0, np.nan)
    
        # note that np.log(nan) = nan
    
        size = np.log(market_cap_at_current_date.astype(float))

        # put the results into the dataframe
    
        if date == listed_stocks.index[0] :
            
            df_size = pd.DataFrame(size.values, index = size.index, columns = [date])
                
        else :
            
            series_size = pd.DataFrame(size.values, index = size.index, columns = [date])
                
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_size = pd.concat([df_size, series_size], axis=1, join='outer')
        
    df_size = df_size.transpose()
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "size.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_size, output)

    output.close()
    
    # print a message 
    
    print('size exposure estimation is done')







def value(): 
    
    ### load fundamental data ###
    
    complete_path = os.path.join(temp_path, "df_fundamental.pkl")
    
    pkfl = open(complete_path,'rb')

    df_fundamental = pickle.load(pkfl)

    pkfl.close()
    
  
    ### total equity ###
    
    total_equity = df_fundamental['total_equity']
  
    ### market cap ###
    
    market_cap = df_fundamental['market_cap']


    # Take the transpose of listed_stocks to facilitate the calculation.

    for stock in market_cap.columns :
        
        # If one of the total_equity or the market_cap is NAN, then the value is NAN.
        
        value = total_equity[stock] / market_cap[stock]

        # put the results into the dataframe
    
        if stock == market_cap.columns[0] :
            
            df_value = pd.DataFrame(value.values, index = market_cap.index)
        
            df_value.columns = [stock]
        

        else :
            
            series_value = pd.DataFrame(value.values, index = market_cap.index)
        
            series_value.columns = [stock]
        
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_value = pd.concat([df_value, series_value], axis=1, join='outer')


    # If the market cap is 0 (why it happens?), then the value is infinite. So replace inf with NAN.

    df_value = df_value.replace(np.inf, np.nan)
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "value.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_value, output)

    output.close()
    
    # print a message 
    
    print('value exposure estimation is done')



            
def leverage():
    
    ### load fundamental data ###
    
    complete_path = os.path.join(temp_path, "df_fundamental.pkl")
    
    pkfl = open(complete_path,'rb')

    df_fundamental = pickle.load(pkfl)

    pkfl.close()
    
  
    ### total equity ###
    
    total_assets = df_fundamental['total_assets']
  
    ### market cap ###
    
    total_liabilities = df_fundamental['total_liabilities']

    
    for stock in total_assets.columns :
        
        # If the total_liabilities or the total_assets is NAN, then the leverage is NAN.
        
        leverage = (total_liabilities[stock] + total_assets[stock]) / total_assets[stock]

        # put the results into the dataframe
    
        if stock == total_assets.columns[0] :
            
            df_leverage = pd.DataFrame(leverage.values, index = total_assets.index)
        
            df_leverage.columns = [stock]
        
        else :
            
            series_leverage = pd.DataFrame(leverage.values, index = total_assets.index)
        
            series_leverage.columns = [stock]
        
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_leverage = pd.concat([df_leverage, series_leverage], axis=1, join='outer')


    # If the market cap is 0 (why it happens?), then the value is infinite. So replace inf with NAN.

    df_leverage = df_leverage.replace(np.inf, np.nan)
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "leverage.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_leverage, output)

    output.close()
    
    # print a message 
    
    print('leverage exposure estimation is done')


            

### function for winsorization and standardization of atomic descriptor ###

def winsorization_and_standardization(df_factor_exposure):
    
    
    #### Standardization ###

    std_factor_exposure = (df_factor_exposure - df_factor_exposure.mean()) / df_factor_exposure.std()

    
    #### Winsorization ###

    sd_factor_exposure = std_factor_exposure.std()
    
    mean_factor_exposure = std_factor_exposure.mean()
    
    upper_limit = mean_factor_exposure + 3*sd_factor_exposure
  
    lower_limit = mean_factor_exposure - 3*sd_factor_exposure
  
    # Replace the outleirs
    
    std_factor_exposure[(std_factor_exposure > upper_limit) & (std_factor_exposure != np.nan)] = upper_limit
    
    std_factor_exposure[(std_factor_exposure < lower_limit) & (std_factor_exposure != np.nan)] = lower_limit
       
    return std_factor_exposure
  
  

def winsorization_and_market_cap_weighed_standardization(df_factor_exposure, market_cap_on_current_day):
    
    
    #### Standardization ###

    std_factor_exposure = (df_factor_exposure - df_factor_exposure.mean()) / df_factor_exposure.std()

    
    #### Market capitalization weighted standardization ###

    market_cap_weighted_factor_exposure = market_cap_on_current_day.multiply(std_factor_exposure, axis='index') 

    market_cap_weighted_factor_exposure_mean = market_cap_weighted_factor_exposure.sum() / market_cap_on_current_day.sum()

    std_market_cap_weighted_factor_exposure = (std_factor_exposure - market_cap_weighted_factor_exposure_mean) / std_factor_exposure.std()
    
    
    #### Winsorization ###

    sd_factor_exposure = std_market_cap_weighted_factor_exposure.std()
    
    mean_factor_exposure = std_market_cap_weighted_factor_exposure.mean()
    
    upper_limit = mean_factor_exposure + 3*sd_factor_exposure
  
    lower_limit = mean_factor_exposure - 3*sd_factor_exposure
  
  
    # Replace the outleirs
    
    std_market_cap_weighted_factor_exposure[(std_market_cap_weighted_factor_exposure > upper_limit) & (std_market_cap_weighted_factor_exposure != np.nan)] = upper_limit
    
    std_market_cap_weighted_factor_exposure[(std_market_cap_weighted_factor_exposure < lower_limit) & (std_market_cap_weighted_factor_exposure != np.nan)] = lower_limit
        
    return std_market_cap_weighted_factor_exposure
  


def two_atomic_descriptors_combination(first_atomic_descriptor_exposure, second_atomic_descriptor_exposure,\
                                       first_atomic_descriptor_weight, second_atomic_descriptor_weight,\
                                       market_cap_on_current_day):

    ### first factor ###

    # obtain the stock list that contains NANs.
    
    first_atomic_descriptor_exposure_nan_stock_list = first_atomic_descriptor_exposure.index[first_atomic_descriptor_exposure.apply(np.isnan)]
    
    # standardization
            
    std_first_atomic_descriptor_exposure = winsorization_and_standardization(first_atomic_descriptor_exposure)
    
    std_market_cap_weighted_first_atomic_descriptor_exposure = winsorization_and_market_cap_weighed_standardization(first_atomic_descriptor_exposure, market_cap_on_current_day)
    
    
    ### second factor ###

    # obtain the stock list that contains NANs.
    
    second_atomic_descriptor_exposure_nan_stock_list = second_atomic_descriptor_exposure.index[second_atomic_descriptor_exposure.apply(np.isnan)]

    # standardization
            
    std_second_atomic_descriptor_exposure = winsorization_and_standardization(second_atomic_descriptor_exposure)
    
    std_market_cap_weighted_second_atomic_descriptor_exposure = winsorization_and_market_cap_weighed_standardization(second_atomic_descriptor_exposure, market_cap_on_current_day)

    
    # combine the atomic descriptors
        
    # if the atomic descriptors exposure contains NANs, replace the NANs with the exposure of other atomic descriptors.

    std_first_atomic_descriptor_exposure.ix[first_atomic_descriptor_exposure_nan_stock_list] = std_second_atomic_descriptor_exposure[first_atomic_descriptor_exposure_nan_stock_list]
    
    std_second_atomic_descriptor_exposure.ix[second_atomic_descriptor_exposure_nan_stock_list] = std_first_atomic_descriptor_exposure[second_atomic_descriptor_exposure_nan_stock_list]

    std_market_cap_weighted_first_atomic_descriptor_exposure.ix[first_atomic_descriptor_exposure_nan_stock_list] = std_market_cap_weighted_second_atomic_descriptor_exposure[first_atomic_descriptor_exposure_nan_stock_list]
    
    std_market_cap_weighted_second_atomic_descriptor_exposure.ix[second_atomic_descriptor_exposure_nan_stock_list] = std_market_cap_weighted_first_atomic_descriptor_exposure[second_atomic_descriptor_exposure_nan_stock_list]
     

    # Use add() function and fill_value = 0, to replace NAN with 0 in part of the atomic descrptors.

    # If all atomic descriptors' exposure are NANs, then the factor exposure is NAN.

    std_factor_exposure = (std_first_atomic_descriptor_exposure*first_atomic_descriptor_weight).add(std_second_atomic_descriptor_exposure*second_atomic_descriptor_weight, fill_value = 0)

    std_market_cap_weighted_factor_exposure = (std_market_cap_weighted_first_atomic_descriptor_exposure*first_atomic_descriptor_weight).add(std_market_cap_weighted_second_atomic_descriptor_exposure*second_atomic_descriptor_weight, fill_value = 0)
 
    
    # restandardization 
 
    std_factor_exposure = winsorization_and_standardization(std_factor_exposure)
    
    std_market_cap_weighted_factor_exposure = winsorization_and_market_cap_weighed_standardization(std_market_cap_weighted_factor_exposure, market_cap_on_current_day)

    return std_factor_exposure, std_market_cap_weighted_factor_exposure




def three_atomic_descriptors_combination(first_atomic_descriptor_exposure, second_atomic_descriptor_exposure, third_atomic_descriptor_exposure,\
                                         first_atomic_descriptor_weight, second_atomic_descriptor_weight, third_atomic_descriptor_weight,\
                                         market_cap_on_current_day):

    ### first factor ###
    
    # obtain the stock list that contains NANs.
    
    first_atomic_descriptor_exposure_nan_stock_list = first_atomic_descriptor_exposure.index[first_atomic_descriptor_exposure.apply(np.isnan)]
    
    # standardization

    std_first_atomic_descriptor_exposure = winsorization_and_standardization(first_atomic_descriptor_exposure)
    
    std_market_cap_weighted_first_atomic_descriptor_exposure = winsorization_and_market_cap_weighed_standardization(first_atomic_descriptor_exposure, market_cap_on_current_day)
 
   
    ### second factor ###
        
    # obtain the stock list that contains NANs.
    
    second_atomic_descriptor_exposure_nan_stock_list = second_atomic_descriptor_exposure.index[second_atomic_descriptor_exposure.apply(np.isnan)]
    
    # standardization
        
    std_second_atomic_descriptor_exposure = winsorization_and_standardization(second_atomic_descriptor_exposure)
    
    std_market_cap_weighted_second_atomic_descriptor_exposure = winsorization_and_market_cap_weighed_standardization(second_atomic_descriptor_exposure, market_cap_on_current_day)


    ### third factor ###

    # obtain the stock list that contains NANs.
    
    third_atomic_descriptor_exposure_nan_stock_list = third_atomic_descriptor_exposure.index[third_atomic_descriptor_exposure.apply(np.isnan)]
    
    # standardization
    
    std_third_atomic_descriptor_exposure = winsorization_and_standardization(third_atomic_descriptor_exposure)
    
    std_market_cap_weighted_third_atomic_descriptor_exposure = winsorization_and_market_cap_weighed_standardization(third_atomic_descriptor_exposure, market_cap_on_current_day)


    # if the atomic descriptors contains NANs, replace the NANs with the values of other atomic descriptors.
  
    # Use add() function and fill_value = 0, to replace NAN with 0 in part of the atomic descrptors.

    std_first_atomic_descriptor_exposure.ix[first_atomic_descriptor_exposure_nan_stock_list] = std_second_atomic_descriptor_exposure[first_atomic_descriptor_exposure_nan_stock_list].add(std_third_atomic_descriptor_exposure[first_atomic_descriptor_exposure_nan_stock_list], fill_value = 0)/2
  
    std_second_atomic_descriptor_exposure.ix[second_atomic_descriptor_exposure_nan_stock_list] = std_first_atomic_descriptor_exposure[second_atomic_descriptor_exposure_nan_stock_list].add(std_third_atomic_descriptor_exposure[second_atomic_descriptor_exposure_nan_stock_list], fill_value = 0)/2

    std_third_atomic_descriptor_exposure.ix[third_atomic_descriptor_exposure_nan_stock_list] = std_first_atomic_descriptor_exposure[third_atomic_descriptor_exposure_nan_stock_list].add(std_second_atomic_descriptor_exposure[third_atomic_descriptor_exposure_nan_stock_list], fill_value = 0)/2
  
  
    std_market_cap_weighted_first_atomic_descriptor_exposure.ix[first_atomic_descriptor_exposure_nan_stock_list] = std_market_cap_weighted_second_atomic_descriptor_exposure[first_atomic_descriptor_exposure_nan_stock_list].add(std_market_cap_weighted_third_atomic_descriptor_exposure[first_atomic_descriptor_exposure_nan_stock_list], fill_value = 0)/2
  
    std_market_cap_weighted_second_atomic_descriptor_exposure.ix[second_atomic_descriptor_exposure_nan_stock_list] = std_market_cap_weighted_first_atomic_descriptor_exposure[second_atomic_descriptor_exposure_nan_stock_list].add(std_market_cap_weighted_third_atomic_descriptor_exposure[second_atomic_descriptor_exposure_nan_stock_list], fill_value = 0)/2

    std_market_cap_weighted_third_atomic_descriptor_exposure.ix[third_atomic_descriptor_exposure_nan_stock_list] = std_market_cap_weighted_first_atomic_descriptor_exposure[third_atomic_descriptor_exposure_nan_stock_list].add(std_market_cap_weighted_second_atomic_descriptor_exposure[third_atomic_descriptor_exposure_nan_stock_list], fill_value = 0)/2


    # If all atomic descriptors' exposure are NANs, then the factor exposure is NAN.

    first_second_combined_exposure = (std_first_atomic_descriptor_exposure*first_atomic_descriptor_weight).add(std_second_atomic_descriptor_exposure*second_atomic_descriptor_weight, fill_value = 0)

    std_factor_exposure = first_second_combined_exposure.add(std_third_atomic_descriptor_exposure*third_atomic_descriptor_weight, fill_value = 0)
    
    market_cap_weighted_first_second_combined_exposure = (std_market_cap_weighted_first_atomic_descriptor_exposure*first_atomic_descriptor_weight).add(std_market_cap_weighted_second_atomic_descriptor_exposure*second_atomic_descriptor_weight, fill_value = 0)

    std_market_cap_weighted_factor_exposure = market_cap_weighted_first_second_combined_exposure.add(std_market_cap_weighted_third_atomic_descriptor_exposure*third_atomic_descriptor_weight, fill_value = 0)
  
    
    # restandardization 
 
    std_factor_exposure = winsorization_and_standardization(std_factor_exposure)
    
    std_market_cap_weighted_factor_exposure = winsorization_and_market_cap_weighed_standardization(std_market_cap_weighted_factor_exposure, market_cap_on_current_day)

    return std_factor_exposure, std_market_cap_weighted_factor_exposure



def factor_exposure_estimation_and_standarization():
    
   
    ### load listed stocks ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks.pkl")
    
    pkfl = open(complete_path,'rb')

    listed_stocks = pickle.load(pkfl)

    pkfl.close()
 
    
    ### load fundamental data ###
    
    complete_path = os.path.join(temp_path, "df_fundamental.pkl")
    
    pkfl = open(complete_path,'rb')

    df_fundamental = pickle.load(pkfl)

    pkfl.close()
    
 
    ### market cap ###
    
    market_cap = df_fundamental['market_cap']
  
  
    ### benchmark beta ###
    
    complete_path = os.path.join(temp_path, "benchmark_beta.pkl")
    
    pkfl = open(complete_path,'rb')

    benchmark_beta = pickle.load(pkfl)

    pkfl.close()
    
    
    ### momentum ###
    
    complete_path = os.path.join(temp_path, "three_month_momentum.pkl")
    
    pkfl = open(complete_path,'rb')

    three_month_momentum = pickle.load(pkfl)

    pkfl.close()
    
    complete_path = os.path.join(temp_path, "six_month_momentum.pkl")
    
    pkfl = open(complete_path,'rb')

    six_month_momentum = pickle.load(pkfl)

    pkfl.close()
    
    
    ### reversal ###
    
    complete_path = os.path.join(temp_path, "reversal.pkl")
    
    pkfl = open(complete_path,'rb')

    reversal = pickle.load(pkfl)

    pkfl.close()


    ### size ###

    complete_path = os.path.join(temp_path, "size.pkl")
    
    pkfl = open(complete_path,'rb')

    size = pickle.load(pkfl)

    pkfl.close()
    
    
    ### earning yield ###
    
    # do the type conversion, otherwise function np.isnan return an error.

    pe_ratio = df_fundamental['pe_ratio'].astype(float)
    
    operating_cash_flow_per_share = df_fundamental['operating_cash_flow_per_share'].astype(float)
    
        
    ### volatility ###
    
    complete_path = os.path.join(temp_path, "short_term_volatility.pkl")
    
    pkfl = open(complete_path,'rb')

    short_term_volatility = pickle.load(pkfl)

    pkfl.close()
    
    complete_path = os.path.join(temp_path, "medium_term_volatility.pkl")
    
    pkfl = open(complete_path,'rb')

    medium_term_volatility = pickle.load(pkfl)

    pkfl.close()
    
    complete_path = os.path.join(temp_path, "long_term_volatility.pkl")
    
    pkfl = open(complete_path,'rb')

    long_term_volatility = pickle.load(pkfl)

    pkfl.close()
    
    
    ### growth ###
    
    # do the type conversion, otherwise function np.isnan return an error.
    
    inc_revenue = df_fundamental['inc_revenue'].astype(float)
    
    inc_total_asset = df_fundamental['inc_total_asset'].astype(float)
    
    inc_gross_profit = df_fundamental['inc_gross_profit'].astype(float)


    ### value ###

    complete_path = os.path.join(temp_path, "value.pkl")
    
    pkfl = open(complete_path,'rb')

    value = pickle.load(pkfl)

    pkfl.close()
    
    
    ### leverage ###

    complete_path = os.path.join(temp_path, "leverage.pkl")
    
    pkfl = open(complete_path,'rb')

    leverage = pickle.load(pkfl)

    pkfl.close()
    
    
    ### liquidity ###
    
    complete_path = os.path.join(temp_path, "short_term_liquidity.pkl")
    
    pkfl = open(complete_path,'rb')

    short_term_liquidity = pickle.load(pkfl)

    pkfl.close()
    
    complete_path = os.path.join(temp_path, "medium_term_liquidity.pkl")
    
    pkfl = open(complete_path,'rb')

    medium_term_liquidity = pickle.load(pkfl)

    pkfl.close()
    
    complete_path = os.path.join(temp_path, "long_term_liquidity.pkl")
    
    pkfl = open(complete_path,'rb')

    long_term_liquidity = pickle.load(pkfl)

    pkfl.close()
    
    
    ### momentum weight ###
    
    complete_path = os.path.join(temp_path, "momentum_weight.pkl")
    
    pkfl = open(complete_path,'rb')

    momentum_weight = pickle.load(pkfl)

    pkfl.close()
    
    
    ### earning yield weight ###
    
    complete_path = os.path.join(temp_path, "earning_yield_weight.pkl")
    
    pkfl = open(complete_path,'rb')

    earning_yield_weight = pickle.load(pkfl)

    pkfl.close()
    
    
    ### earning yield weight ###
    
    complete_path = os.path.join(temp_path, "volatility_weight.pkl")
    
    pkfl = open(complete_path,'rb')

    volatility_weight = pickle.load(pkfl)

    pkfl.close()

    
    ### growth weight ###
    
    complete_path = os.path.join(temp_path, "growth_weight.pkl")
    
    pkfl = open(complete_path,'rb')

    growth_weight = pickle.load(pkfl)

    pkfl.close()
    
    
    ### liquidity weight ###
    
    complete_path = os.path.join(temp_path, "liquidity_weight.pkl")
    
    pkfl = open(complete_path,'rb')

    liquidity_weight = pickle.load(pkfl)

    pkfl.close()
    
    
    # Take the transpose of listed_stocks to facilitate the calculation.

    listed_stocks_t = listed_stocks.transpose()
    
    
    # skip the first 5 trading days, on which we don't need to estimate factor exposure  
 
    for date in listed_stocks.index[5:] : 
        
        #print(date)    
    
        # Obtain the order_book_id list of listed stock today and yesterday.

        list = listed_stocks_t[listed_stocks_t[date] == 'True'].index.tolist()
   
        market_cap_on_current_day = market_cap[list].ix[date].copy()
        
        # Take the average of the past 5 trading days as today's weight

        five_day_before = listed_stocks.index.get_loc(date) - 4
    
        five_day_before = listed_stocks.index[five_day_before]

    
    
        ##### BETA FACTOR ######

        # Make a deep copy of a slice of the dataframe

        benchmark_beta_on_current_day = benchmark_beta[list].ix[date].copy() 
            
        std_benchmark_beta = winsorization_and_standardization(benchmark_beta_on_current_day)
    
        std_market_cap_weighted_benchmark_beta = winsorization_and_market_cap_weighed_standardization(benchmark_beta_on_current_day, market_cap_on_current_day)
  

    
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_benchmark_beta = pd.DataFrame(std_benchmark_beta.values, index = std_benchmark_beta.index, columns = [date])
   
            df_std_market_cap_weighted_benchmark_beta = pd.DataFrame(std_market_cap_weighted_benchmark_beta.values, index = std_market_cap_weighted_benchmark_beta.index, columns = [date])
             
        else:
            
            series_std_benchmark_beta = pd.DataFrame(std_benchmark_beta.values, index = std_benchmark_beta.index, columns = [date])
 
            series_std_market_cap_weighted_benchmark_beta = pd.DataFrame(std_market_cap_weighted_benchmark_beta.values, index = std_market_cap_weighted_benchmark_beta.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_benchmark_beta = pd.concat([df_std_benchmark_beta, series_std_benchmark_beta], axis=1, join='outer')

            df_std_market_cap_weighted_benchmark_beta = pd.concat([df_std_market_cap_weighted_benchmark_beta, series_std_market_cap_weighted_benchmark_beta], axis=1, join='outer')
 
 
        ###### MOMENTUM FACTOR ######

        ### three month momentum ###

        three_month_momentum_on_current_day = three_month_momentum[list].ix[date].copy() 
      
    
        ### six month momentum ###

        six_month_momentum_on_current_day = six_month_momentum[list].ix[date].copy()   
  
       
        # atomic descriptors' weight on current trading day is computed as the mean of ones over the last 5 trading days.
       
        three_month_momentum_weight = momentum_weight['three_month_momentum_weight'].ix[five_day_before:date].mean()

        six_month_momentum_weight = momentum_weight['six_month_momentum_weight'].ix[five_day_before:date].mean()
    
    
        std_momentum, std_market_cap_weighted_momentum\
        =  two_atomic_descriptors_combination(three_month_momentum_on_current_day, six_month_momentum_on_current_day,\
                                              three_month_momentum_weight, six_month_momentum_weight,\
                                              market_cap_on_current_day)

 
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_momentum = pd.DataFrame(std_momentum.values, index = std_momentum.index, columns = [date])
   
            df_std_market_cap_weighted_momentum = pd.DataFrame(std_market_cap_weighted_momentum.values, index = std_market_cap_weighted_momentum.index, columns = [date])
             
        else :
            
            series_std_momentum = pd.DataFrame(std_momentum.values, index = std_momentum.index, columns = [date])
 
            series_std_market_cap_weighted_momentum = pd.DataFrame(std_market_cap_weighted_momentum.values, index = std_market_cap_weighted_momentum.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_momentum = pd.concat([df_std_momentum, series_std_momentum], axis=1, join='outer')

            df_std_market_cap_weighted_momentum = pd.concat([df_std_market_cap_weighted_momentum, series_std_market_cap_weighted_momentum], axis=1, join='outer')
 
 
    
        ##### REVERSAL FACTOR ######

        # Make a deep copy of a slice of the dataframe

        reversal_on_current_day = reversal[list].ix[date].copy() 
            
        std_reversal = winsorization_and_standardization(reversal_on_current_day)
    
        std_market_cap_weighted_reversal = winsorization_and_market_cap_weighed_standardization(reversal_on_current_day, market_cap_on_current_day)
    
         
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_reversal = pd.DataFrame(std_reversal.values, index = std_reversal.index, columns = [date])
   
            df_std_market_cap_weighted_reversal = pd.DataFrame(std_market_cap_weighted_reversal.values, index = std_market_cap_weighted_reversal.index, columns = [date])
             
        else :
            
            series_std_reversal = pd.DataFrame(std_reversal.values, index = std_reversal.index, columns = [date])
 
            series_std_market_cap_weighted_reversal = pd.DataFrame(std_market_cap_weighted_reversal.values, index = std_market_cap_weighted_reversal.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_reversal = pd.concat([df_std_reversal, series_std_reversal], axis=1, join='outer')

            df_std_market_cap_weighted_reversal = pd.concat([df_std_market_cap_weighted_reversal, series_std_market_cap_weighted_reversal], axis=1, join='outer')



        ##### SIZE FACTOR ######

        # Make a deep copy of a slice of the dataframe

        size_on_current_day = size[list].ix[date].copy() 
            
        std_size = winsorization_and_standardization(size_on_current_day)
    
        std_market_cap_weighted_size = winsorization_and_market_cap_weighed_standardization(size_on_current_day, market_cap_on_current_day)
    
         
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_size = pd.DataFrame(std_size.values, index = std_size.index, columns = [date])
   
            df_std_market_cap_weighted_size = pd.DataFrame(std_market_cap_weighted_size.values, index = std_market_cap_weighted_size.index, columns = [date])
             
        else :
            
            series_std_size = pd.DataFrame(std_size.values, index = std_market_cap_weighted_size.index, columns = [date])
 
            series_std_market_cap_weighted_size = pd.DataFrame(std_market_cap_weighted_size.values, index = std_market_cap_weighted_size.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_size = pd.concat([df_std_size, series_std_size], axis=1, join='outer')

            df_std_market_cap_weighted_size = pd.concat([df_std_market_cap_weighted_size, series_std_market_cap_weighted_size], axis=1, join='outer')
 

 
        ###### EARNING YIELD FACTOR ######


        ### pe ratio ###

        pe_ratio_on_current_day = pe_ratio[list].ix[date].copy() 
    
    
        ### operating_cash_flow_per_share ###    
    
        operating_cash_flow_per_share_on_current_day = operating_cash_flow_per_share[list].ix[date].copy() 
    
        
        # Take the average of the past 5 trading days as today's weight
    
        pe_ratio_weight = earning_yield_weight['pe_ratio_weight'].ix[five_day_before:date].mean()

        operating_cash_flow_per_share_weight = earning_yield_weight['operating_cash_flow_per_share_weight'].ix[five_day_before:date].mean()

        
        std_earning_yield, std_market_cap_weighted_earning_yield\
        =  two_atomic_descriptors_combination(pe_ratio_on_current_day, operating_cash_flow_per_share_on_current_day,\
                                              pe_ratio_weight, operating_cash_flow_per_share_weight,\
                                              market_cap_on_current_day)

 
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_earning_yield = pd.DataFrame(std_earning_yield.values, index = std_earning_yield.index, columns = [date])
   
            df_std_market_cap_weighted_earning_yield = pd.DataFrame(std_market_cap_weighted_earning_yield.values, index = std_market_cap_weighted_earning_yield.index, columns = [date])
             
        else :
            
            series_std_earning_yield = pd.DataFrame(std_earning_yield.values, index = std_earning_yield.index, columns = [date])
 
            series_std_market_cap_weighted_earning_yield = pd.DataFrame(std_market_cap_weighted_earning_yield.values, index = std_market_cap_weighted_earning_yield.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_earning_yield = pd.concat([df_std_earning_yield, series_std_earning_yield], axis=1, join='outer')

            df_std_market_cap_weighted_earning_yield = pd.concat([df_std_market_cap_weighted_earning_yield, series_std_market_cap_weighted_earning_yield], axis=1, join='outer')
 

    
        ###### VOLATILITY FACTOR ######

        ### short term volatility ###

        short_term_volatility_on_current_day = short_term_volatility[list].ix[date].copy() 
        
    
        ### medium term volatility ###

        medium_term_volatility_on_current_day = medium_term_volatility[list].ix[date].copy() 
    

        ### long term volatility ###

        long_term_volatility_on_current_day = long_term_volatility[list].ix[date].copy() 
  
  
        # obtain the average of atomic descriptors' weight
    
        short_term_volatility_weight = volatility_weight['short_term_volatility_weight'].ix[five_day_before:date].mean()

        medium_term_volatility_weight = volatility_weight['medium_term_volatility_weight'].ix[five_day_before:date].mean()

        long_term_volatility_weight = volatility_weight['long_term_volatility_weight'].ix[five_day_before:date].mean()


        std_volatility, std_market_cap_weighted_volatility\
        = three_atomic_descriptors_combination(short_term_volatility_on_current_day, medium_term_volatility_on_current_day, long_term_volatility_on_current_day,\
                                               short_term_volatility_weight, medium_term_volatility_weight, long_term_volatility_weight,\
                                               market_cap_on_current_day)

 
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_volatility = pd.DataFrame(std_volatility.values, index = std_volatility.index, columns = [date])
   
            df_std_market_cap_weighted_volatility = pd.DataFrame(std_market_cap_weighted_volatility.values, index = std_market_cap_weighted_volatility.index, columns = [date])
             
        else :
            
            series_std_volatility = pd.DataFrame(std_volatility.values, index = std_volatility.index, columns = [date])
 
            series_std_market_cap_weighted_volatility = pd.DataFrame(std_market_cap_weighted_volatility.values, index = std_market_cap_weighted_volatility.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_volatility = pd.concat([df_std_volatility, series_std_volatility], axis=1, join='outer')

            df_std_market_cap_weighted_volatility = pd.concat([df_std_market_cap_weighted_volatility, series_std_market_cap_weighted_volatility], axis=1, join='outer')
 
    

        ###### GROWTH FACTOR ######

        ### inc_revenue ###

        inc_revenue_on_current_day = inc_revenue[list].ix[date].copy() 
    
    
        ### inc_total_asset ###

        inc_total_asset_on_current_day =inc_total_asset[list].ix[date].copy() 
    

        ### inc_gross_profit ###

        inc_gross_profit_on_current_day = inc_gross_profit[list].ix[date].copy() 
    
    
        # Take the average of the past 5 trading days as today's weight
    
        inc_revenue_weight = growth_weight['inc_revenue_weight'].ix[five_day_before:date].mean()

        inc_total_asset_weight = growth_weight['inc_total_asset_weight'].ix[five_day_before:date].mean()

        inc_gross_profit_weight = growth_weight['inc_gross_profit_weight'].ix[five_day_before:date].mean()
    
        std_growth, std_market_cap_weighted_growth\
        = three_atomic_descriptors_combination(inc_revenue_on_current_day, inc_total_asset_on_current_day, inc_gross_profit_on_current_day,\
                                               inc_revenue_weight, inc_total_asset_weight, inc_gross_profit_weight,\
                                               market_cap_on_current_day)


 
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_growth = pd.DataFrame(std_growth.values, index = std_growth.index, columns = [date])
   
            df_std_market_cap_weighted_growth = pd.DataFrame(std_market_cap_weighted_growth.values, index = std_market_cap_weighted_growth.index, columns = [date])
             
        else :
                
            series_std_growth = pd.DataFrame(std_growth.values, index = std_growth.index, columns = [date])
 
            series_std_market_cap_weighted_growth = pd.DataFrame(std_market_cap_weighted_growth.values, index = std_market_cap_weighted_growth.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_growth = pd.concat([df_std_growth, series_std_growth], axis=1, join='outer')

            df_std_market_cap_weighted_growth = pd.concat([df_std_market_cap_weighted_growth, series_std_market_cap_weighted_growth], axis=1, join='outer')
 
 
 
        ###### VALUE FACTOR ######

        # Make a deep copy of a slice of the dataframe

        value_on_current_day = value[list].ix[date].copy() 
            
        std_value = winsorization_and_standardization(value_on_current_day)
    
        std_market_cap_weighted_value = winsorization_and_market_cap_weighed_standardization(value_on_current_day, market_cap_on_current_day)
    
        
    
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
                
            df_std_value = pd.DataFrame(std_value.values, index = std_value.index, columns = [date])
   
            df_std_market_cap_weighted_value = pd.DataFrame(std_market_cap_weighted_value.values, index = std_market_cap_weighted_value.index, columns = [date])
             
        else :
            
            series_std_value = pd.DataFrame(std_value.values, index = std_value.index, columns = [date])
 
            series_std_market_cap_weighted_value = pd.DataFrame(std_market_cap_weighted_value.values, index = std_market_cap_weighted_value.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_value = pd.concat([df_std_value, series_std_value], axis=1, join='outer')

            df_std_market_cap_weighted_value = pd.concat([df_std_market_cap_weighted_value, series_std_market_cap_weighted_value], axis=1, join='outer')
 
 
 
        ###### LEVERAGE FACTOR ######

        # Make a deep copy of a slice of the dataframe

        leverage_on_current_day = leverage[list].ix[date].copy() 
            
        std_leverage = winsorization_and_standardization(leverage_on_current_day)
    
        std_market_cap_weighted_leverage = winsorization_and_market_cap_weighed_standardization(leverage_on_current_day, market_cap_on_current_day)
        
    
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_leverage = pd.DataFrame(std_leverage.values, index = std_leverage.index, columns = [date])
   
            df_std_market_cap_weighted_leverage = pd.DataFrame(std_market_cap_weighted_leverage.values, index = std_market_cap_weighted_leverage.index, columns = [date])
             
        else :
            
            series_std_leverage = pd.DataFrame(std_leverage.values, index = std_leverage.index, columns = [date])
 
            series_std_market_cap_weighted_leverage = pd.DataFrame(std_market_cap_weighted_leverage.values, index = std_market_cap_weighted_leverage.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_leverage = pd.concat([df_std_leverage, series_std_leverage], axis=1, join='outer')

            df_std_market_cap_weighted_leverage = pd.concat([df_std_market_cap_weighted_leverage, series_std_market_cap_weighted_leverage], axis=1, join='outer')


        ###### LIQUIDITY FACTOR ######

        ### short term liquidity ###

        short_term_liquidity_on_current_day = short_term_liquidity[list].ix[date].copy() 
        
    
        ### medium term liquidity ###

        medium_term_liquidity_on_current_day = medium_term_liquidity[list].ix[date].copy() 
    

        #### long term liquidity ###

        long_term_liquidity_on_current_day = long_term_liquidity[list].ix[date].copy() 
    
    
        # Take the average of the past 5 trading days as today's weight
    
        short_term_liquidity_weight = liquidity_weight['short_term_liquidity_weight'].ix[five_day_before:date].mean()

        medium_term_liquidity_weight = liquidity_weight['medium_term_liquidity_weight'].ix[five_day_before:date].mean()

        long_term_liquidity_weight = liquidity_weight['long_term_liquidity_weight'].ix[five_day_before:date].mean()
    

        std_liquidity, std_market_cap_weighted_liquidity\
        = three_atomic_descriptors_combination(short_term_liquidity_on_current_day, medium_term_liquidity_on_current_day, long_term_liquidity_on_current_day,\
                                               short_term_liquidity_weight, medium_term_liquidity_weight, long_term_liquidity_weight,\
                                               market_cap_on_current_day)

 
        # put the results into a dataframe
    
        if date == listed_stocks.index[5] :
            
            df_std_liquidity = pd.DataFrame(std_liquidity.values, index = std_liquidity.index, columns = [date])
   
            df_std_market_cap_weighted_liquidity = pd.DataFrame(std_market_cap_weighted_liquidity.values, index = std_market_cap_weighted_liquidity.index, columns = [date])
             
        else :
            
            series_std_liquidity = pd.DataFrame(std_liquidity.values, index = std_liquidity.index, columns = [date])
 
            series_std_market_cap_weighted_liquidity = pd.DataFrame(std_market_cap_weighted_liquidity.values, index = std_market_cap_weighted_liquidity.index, columns = [date])
       
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_std_liquidity = pd.concat([df_std_liquidity, series_std_liquidity], axis=1, join='outer')

            df_std_market_cap_weighted_liquidity = pd.concat([df_std_market_cap_weighted_liquidity, series_std_market_cap_weighted_liquidity], axis=1, join='outer')
 
    

    df_std_benchmark_beta = df_std_benchmark_beta.transpose()

    df_std_market_cap_weighted_benchmark_beta = df_std_market_cap_weighted_benchmark_beta.transpose()


    df_std_momentum = df_std_momentum.transpose()

    df_std_market_cap_weighted_momentum = df_std_market_cap_weighted_momentum.transpose()


    df_std_reversal = df_std_reversal.transpose()

    df_std_market_cap_weighted_reversal = df_std_market_cap_weighted_reversal.transpose()


    df_std_size = df_std_size.transpose()

    df_std_market_cap_weighted_size = df_std_market_cap_weighted_size.transpose()


    df_std_earning_yield = df_std_earning_yield.transpose()

    df_std_market_cap_weighted_earning_yield = df_std_market_cap_weighted_earning_yield.transpose()


    df_std_volatility = df_std_volatility.transpose()

    df_std_market_cap_weighted_volatility = df_std_market_cap_weighted_volatility.transpose()


    df_std_growth = df_std_growth.transpose()

    df_std_market_cap_weighted_growth = df_std_market_cap_weighted_growth.transpose()


    df_std_value = df_std_value.transpose()

    df_std_market_cap_weighted_value = df_std_market_cap_weighted_value.transpose()


    df_std_leverage = df_std_leverage.transpose()

    df_std_market_cap_weighted_leverage = df_std_market_cap_weighted_leverage.transpose()


    df_std_liquidity = df_std_liquidity.transpose()

    df_std_market_cap_weighted_liquidity = df_std_market_cap_weighted_liquidity.transpose()
    
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "std_benchmark_beta.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_benchmark_beta, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_benchmark_beta.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_benchmark_beta, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_momentum.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_momentum, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_momentum.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_momentum, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_reversal.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_reversal, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_reversal.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_reversal, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_size.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_size, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_size.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_size, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_earning_yield.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_earning_yield, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_earning_yield.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_earning_yield, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_volatility.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_volatility, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_volatility.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_volatility, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_growth.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_growth, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_growth.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_growth, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_value.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_value, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_value.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_value, output)

    output.close()
        
    
    complete_path = os.path.join(temp_path, "std_leverage.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_leverage, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_leverage.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_leverage, output)

    output.close()
    
    
    complete_path = os.path.join(temp_path, "std_liquidity.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_liquidity, output)

    output.close()
    
    complete_path = os.path.join(temp_path, "std_market_cap_weighted_liquidity.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_std_market_cap_weighted_liquidity, output)

    output.close()
    
    # print a message 
    
    print('factor exposure estimation and standarization is done')







def style_factor_exposure():
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    
    print('style factor exposure estimation begins')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    
    size()
    
    value()
    
    leverage()
    
    factor_exposure_estimation_and_standarization()
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    
    print('style factor exposure estimation is done')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    
    





