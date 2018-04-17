

###### 缺失值估计和填补 ######


### 模块说明 ###

# 本模块实现缺失值的估计和填补。

# 在 RQBeta 的数据处理流程中，共有五个步骤，其中最后两个步骤在本模块完成：

#（1） pe_ratio 缺失值用 0 替换（数据读取时完成）；

#（2）量价因子的邻近交易日填补后计算 (在 atomic_descriptors完成）；

#（3）细分因子缺失值填补 (在 atomic_descriptors完成）；

#（4）邻近交易日缺失值填补；

#（5）缺失值的回归估计。



###### 输出数据 ######

# 缺失值被填补的风格暴露度 dataframe；


###### 工程备忘 ######

### 1 在计算过程中，尚未发现量价因子需要进行缺失值估计；需要进行缺失值估计的情况较多出现在价值因子和杠杆因子；

### 2 在剔除上市未足半年及 ST 股后，发现仍有部分股票没有申万行业标记，对缺失值回归估计造成一定的影响；

### 3 风格暴露度既是下一步计算因子收益的中间数据，也是最终结果，因此计算结果同时保存在临时文件夹，及计算结果文件夹（临时文件夹部分多一个交易日的数据，用于因子收益计算）。




### import external packages ###

import numpy as np
import pandas as pd
import statsmodels.api as sm

import datetime
import pickle
import os.path


### paths for saving files ###

temp_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/temp/"

results_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/results/"



def style_factor_exposure_imputation():
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    
    print('style factor exposure imputation begins')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    
    ### load stocks listed for 133 trading days ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks_for_133_trading_days.pkl")
    
    pkfl = open(complete_path,'rb')

    listed_stocks_for_133_trading_days = pickle.load(pkfl)

    pkfl.close()
    
    
    ### load st stocks ###
    
    complete_path = os.path.join(temp_path, "st_stocks.pkl")
    
    pkfl = open(complete_path,'rb')

    st_stocks = pickle.load(pkfl)

    pkfl.close()
    
    
    ### load stocks industry classification ###
    
    complete_path = os.path.join(temp_path, "stocks_industry_classification.pkl")
    
    pkfl = open(complete_path,'rb')

    stocks_industry_classification = pickle.load(pkfl)

    pkfl.close()
    
    
    ### std_benchmark_beta ###
    
    complete_path = os.path.join(temp_path, "std_benchmark_beta.pkl")
    
    pkfl = open(complete_path,'rb')

    std_benchmark_beta = pickle.load(pkfl)

    pkfl.close()


    ### std_momentum ###

    complete_path = os.path.join(temp_path, "std_momentum.pkl")
    
    pkfl = open(complete_path,'rb')

    std_momentum = pickle.load(pkfl)

    pkfl.close()


    ### std_reversal ###

    complete_path = os.path.join(temp_path, "std_reversal.pkl")
    
    pkfl = open(complete_path,'rb')

    std_reversal = pickle.load(pkfl)

    pkfl.close()


    ### std_size ###

    complete_path = os.path.join(temp_path, "std_size.pkl")
    
    pkfl = open(complete_path,'rb')

    std_size = pickle.load(pkfl)

    pkfl.close()


    ### std_earning_yield ###

    complete_path = os.path.join(temp_path, "std_earning_yield.pkl")
    
    pkfl = open(complete_path,'rb')

    std_earning_yield = pickle.load(pkfl)

    pkfl.close()


    ### std_volatility ###

    complete_path = os.path.join(temp_path, "std_volatility.pkl")
    
    pkfl = open(complete_path,'rb')

    std_volatility = pickle.load(pkfl)

    pkfl.close()


    ### std_growth ###

    complete_path = os.path.join(temp_path, "std_growth.pkl")
    
    pkfl = open(complete_path,'rb')

    std_growth = pickle.load(pkfl)

    pkfl.close()


    ### std_value ###

    complete_path = os.path.join(temp_path, "std_value.pkl")
    
    pkfl = open(complete_path,'rb')

    std_value = pickle.load(pkfl)

    pkfl.close()


    ### std_leverage ###

    complete_path = os.path.join(temp_path, "std_leverage.pkl")
    
    pkfl = open(complete_path,'rb')

    std_leverage = pickle.load(pkfl)

    pkfl.close()


    ### std_liquidity ###

    complete_path = os.path.join(temp_path, "std_liquidity.pkl")
    
    pkfl = open(complete_path,'rb')

    std_liquidity = pickle.load(pkfl)

    pkfl.close()


    ### Intialize dataframes ###

    # skip the first 10 trading days, they are computed merely for missing data imputation.

    std_benchmark_beta_missing_data_imputed = std_benchmark_beta[10:].copy()

    std_momentum_missing_data_imputed = std_momentum[10:].copy()
    
    std_reversal_missing_data_imputed = std_reversal[10:].copy()

    std_size_missing_data_imputed = std_size[10:].copy()

    std_earning_yield_missing_data_imputed = std_earning_yield[10:].copy()

    std_volatility_missing_data_imputed = std_volatility[10:].copy()

    std_growth_missing_data_imputed = std_growth[10:].copy()

    std_value_missing_data_imputed = std_value[10:].copy()

    std_leverage_missing_data_imputed = std_leverage[10:].copy()

    std_liquidity_missing_data_imputed = std_liquidity[10:].copy()


    # take the transpose to faciliate the calculation 

    listed_stocks_for_133_trading_days_t = listed_stocks_for_133_trading_days.transpose()

    st_stock_t = st_stocks.transpose()


    # skip the first 15 trading days.

    for date in listed_stocks_for_133_trading_days.index[16:] :
        
        #print(date)
    
        # Obtain the order_book_id list of stocks that are listed for more than 132 trading days as well as not "ST" at current trading day.
   
        # Qualified stocks are labelled as 1 in the dataframe.
   
        listed_stock_list = listed_stocks_for_133_trading_days_t[listed_stocks_for_133_trading_days_t[date] == 'True'].index.tolist()
    
        # Qualified stocks are labelled as 0 in the dataframe.
    
        non_st_stock_list = st_stock_t[st_stock_t[date] == 'False'].index.tolist()
    
        combined_list = [x for x in listed_stock_list if x in non_st_stock_list]

        # Obtain the industry label of stocks at current date

        stocks_industry_classification_on_current_day = stocks_industry_classification[combined_list].ix[date]
    
    
        # NAN in benchmark beta factor
    
        benchmark_beta_on_current_day = std_benchmark_beta[combined_list].ix[date]
    
        benchmark_beta_nan_stock_list = benchmark_beta_on_current_day.index[benchmark_beta_on_current_day.apply(np.isnan)]

    
        # NAN in momentum factor
    
        momentum_on_current_day = std_momentum[combined_list].ix[date]
    
        momentum_nan_stock_list = momentum_on_current_day.index[momentum_on_current_day.apply(np.isnan)]

    
        # NAN in reversal factor
    
        reversal_on_current_day = std_reversal[combined_list].ix[date]
    
        reversal_nan_stock_list = reversal_on_current_day.index[reversal_on_current_day.apply(np.isnan)]


        # NAN in size factor
    
        size_on_current_day = std_size[combined_list].ix[date]
    
        size_nan_stock_list = size_on_current_day.index[size_on_current_day.apply(np.isnan)]

 
        # NAN in earing yield factor
    
        earning_yield_on_current_day = std_earning_yield[combined_list].ix[date]
    
        earning_yield_nan_stock_list = earning_yield_on_current_day.index[earning_yield_on_current_day.apply(np.isnan)]


        # NAN in volatility factor
    
        volatility_on_current_day = std_volatility[combined_list].ix[date]
    
        volatility_nan_stock_list = volatility_on_current_day.index[volatility_on_current_day.apply(np.isnan)]
    
    
        # NAN in growth factor
    
        growth_on_current_day = std_growth[combined_list].ix[date]
    
        growth_nan_stock_list = growth_on_current_day.index[growth_on_current_day.apply(np.isnan)]


        # NAN in value factor
    
        value_on_current_day = std_value[combined_list].ix[date]
    
        value_nan_stock_list = value_on_current_day.index[value_on_current_day.apply(np.isnan)]


        # NAN in leverage factor
    
        leverage_on_current_day = std_leverage[combined_list].ix[date]
    
        leverage_nan_stock_list = leverage_on_current_day.index[leverage_on_current_day.apply(np.isnan)]


        # NAN in liquidity factor
    
        liquidity_on_current_day = std_liquidity[combined_list].ix[date]
    
        liquidity_nan_stock_list = liquidity_on_current_day.index[liquidity_on_current_day.apply(np.isnan)]
        

        nan_combined_list = [benchmark_beta_nan_stock_list, momentum_nan_stock_list, reversal_nan_stock_list,\
                             size_nan_stock_list, earning_yield_nan_stock_list, volatility_nan_stock_list,\
                             growth_nan_stock_list, value_nan_stock_list, leverage_nan_stock_list,\
                             liquidity_nan_stock_list]
          
               
        # Create a list of stocks that have NANs in factor exposures.
        
        factors_exposure_missing_stock_list = set.union(*map(set, nan_combined_list))

        # Put all factor exposure into one dataframe
                         
        factor_exposure = pd.concat([benchmark_beta_on_current_day, momentum_on_current_day, reversal_on_current_day,\
                                     size_on_current_day, earning_yield_on_current_day, volatility_on_current_day,\
                                     growth_on_current_day, value_on_current_day, leverage_on_current_day,\
                                     liquidity_on_current_day], axis=1)
        
                                 
        factor_exposure.columns = ['benchmark_beta', 'momentum', 'reversal', 'size', 'earning_yield', 'volatility', 'growth', 'value', 'leverage', 'liquidity'] 
    
        # Add an intercept for the regression estimation.
    
        factor_exposure = sm.add_constant(factor_exposure, prepend=False)
       
    
        ###### benchmark beta exposure imputation ######
    
        if len(nan_combined_list[0]) != 0 :
            
            print('numbers of benchmark_beta factor exposure missing', len(nan_combined_list[0]))
 
            today = std_benchmark_beta.index.get_loc(date) 
            eleven_days_before = std_benchmark_beta.index.get_loc(date) - 10
        
            for stock in benchmark_beta_nan_stock_list :
                
                # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
 
                mean_of_past_10_days = std_benchmark_beta[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :
                    
                    # Obtain the industry index of the stock 
                    
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry, as well as no factor expsoure missing in it.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for estimation of missing factor exposure
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the benchmark beta exposure against all the other factors' exposure

                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 0], estimation_factor_exposure.ix[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).fit()
            
                    # Predict the benchmark beta exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposures with NAN is ignored in prediction.

                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].fillna(0))
            
                    std_benchmark_beta_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure
               
                else :
                    
                    std_benchmark_beta_missing_data_imputed[stock].ix[date] = mean_of_past_10_days
                

        ###### momentum exposure imputation ######
    
        if len(nan_combined_list[1]) != 0 :
            
            print('numbers of momentum factor exposure missing', len(nan_combined_list[1]))

        
            today = std_momentum.index.get_loc(date) 
            eleven_days_before = std_momentum.index.get_loc(date) - 10
        
    
            for stock in momentum_nan_stock_list :
                
                # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
            
                mean_of_past_10_days = std_momentum[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :
                    
                    # obtain the industry index of the stock 

                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the momentum exposure against all the other factors' exposure
        
                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 1], estimation_factor_exposure.ix[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).fit()
            
                    # Predict the momentum exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ignored in prediction.

                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]].fillna(0))
            
                    std_momentum_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure

               
                else :
                         
                    std_momentum_missing_data_imputed[stock].ix[date] = mean_of_past_10_days
          
    
        ###### reversal exposure imputation ######
    
        if len(nan_combined_list[2]) != 0 :
            
            print('numbers of reversal factor exposure missing', len(nan_combined_list[2]))

        
            today = std_reversal.index.get_loc(date) 
            eleven_days_before = std_reversal.index.get_loc(date) - 10
        
    
            for stock in reversal_nan_stock_list :
                
                # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
                
                mean_of_past_10_days = std_reversal[stock].ix[eleven_days_before:today].mean()
            
            
                if np.isnan(mean_of_past_10_days) :
                    
                    # obtain the industry index of the stock

                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the reversal exposure against all the other factors' exposure

                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 2], estimation_factor_exposure.ix[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]]).fit()
            
                    # Predict the reversal exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.

                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]].fillna(0))
            
                    std_reversal_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure
 
               
                else :

                    std_reversal_missing_data_imputed[stock].ix[date] = mean_of_past_10_days
            

        ###### size exposure imputation ######
    
        if len(nan_combined_list[3]) != 0 :
            
            print('numbers of size factor exposure missing', len(nan_combined_list[3]))

       
            today = std_size.index.get_loc(date) 
            eleven_days_before = std_size.index.get_loc(date) - 10
        
    
            for stock in size_nan_stock_list :
            
                # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
             
                mean_of_past_10_days = std_size[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :

                    # obtain the industry index of the stock 
            
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the size exposure against all the other factors' exposure

                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 3], estimation_factor_exposure.ix[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]).fit()
            
                    # Predict the size exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.

                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]].fillna(0))
            
                    std_size_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure

               
                else :
                         
                    std_size_missing_data_imputed[stock].ix[date] = mean_of_past_10_days


        ###### earning yield exposure imputation ######
    
        if len(nan_combined_list[4]) != 0 :
        
            print('numbers of earning_yield factor exposure missing', len(nan_combined_list[4]))
      
            today = std_earning_yield.index.get_loc(date) 
            eleven_days_before = std_earning_yield.index.get_loc(date) - 10
        
    
            for stock in earning_yield_nan_stock_list :
            
               # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
             
                mean_of_past_10_days = std_earning_yield[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :

                    # obtain the industry index of the stock
            
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the earning_yield exposure against all the other factors' exposure
   
                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 4], estimation_factor_exposure.ix[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]]).fit()
            
                    # Predict the earning_yield exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.

                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]].fillna(0))
            
                    std_earning_yield_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure
           
                else :
                         
                    std_earning_yield_missing_data_imputed[stock].ix[date] = mean_of_past_10_days
          
  
            ###### volatility exposure estimation ######
    
            if len(nan_combined_list[5]) != 0 :
        
                print('numbers of volatility factor exposure missing', len(nan_combined_list[5]))
     
                today = std_volatility.index.get_loc(date) 
                eleven_days_before = std_volatility.index.get_loc(date) - 10
        
    
                for stock in volatility_nan_stock_list :
            
                    # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
             
                    mean_of_past_10_days = std_volatility[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :

                    # obtain the industry index of the stock 
            
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the volatility exposure against all the other factors' exposure

                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 5], estimation_factor_exposure.ix[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]]).fit()
            
                    # Predict the volatility exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.

                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]].fillna(0))
            
                    std_volatility_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure

               
                else :
                         
                    std_volatility_missing_data_imputed[stock].ix[date] = mean_of_past_10_days


        ###### growth exposure imputation ######
    
        if len(nan_combined_list[6]) != 0 :
        
            print('numbers of growth factor exposure missing', len(nan_combined_list[6]))
      
            today = std_growth.index.get_loc(date)
            eleven_days_before = std_growth.index.get_loc(date) - 10 
    
            for stock in growth_nan_stock_list :
            
                # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
             
                mean_of_past_10_days = std_growth[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :

                    # obtain the industry index of the stock
           
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the growth exposure against all the other factors' exposure

                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 6], estimation_factor_exposure.ix[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]]).fit()
            
                    # Predict the growth exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.
            
                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]].fillna(0))
            
                    std_growth_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure
             
                else :
                         
                    std_growth_missing_data_imputed[stock].ix[date] = mean_of_past_10_days

                
        ###### value exposure imputation ######
    
        if len(nan_combined_list[7]) != 0 :
        
            print('numbers of value factor exposure missing', len(nan_combined_list[7]))

        
            today = std_value.index.get_loc(date)
            eleven_days_before = std_value.index.get_loc(date) - 10
        
    
            for stock in value_nan_stock_list :
            
                # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
            
                mean_of_past_10_days = std_value[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :

                    # obtain the industry index of the stock 
            
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the value exposure against all the other factors' exposure

                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.
            
                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 7], estimation_factor_exposure.ix[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]].fillna(0)).fit()
            
                    # Predict the value exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.
           
                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]].fillna(0))
            
                    std_value_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure

               
                else :
                
                    std_value_missing_data_imputed[stock].ix[date] = mean_of_past_10_days


        ###### leverage exposure imputation ######
    
        if len(nan_combined_list[8]) != 0 :
        
            print('numbers of leverage factor exposure missing', len(nan_combined_list[8]))

        
            today = std_leverage.index.get_loc(date)
            eleven_days_before = std_leverage.index.get_loc(date) - 10
        
    
            for stock in leverage_nan_stock_list :

               # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
             
                mean_of_past_10_days = std_leverage[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :
  
                    # obtain the industry index of the stock 
           
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the leverage exposure against all the other factors' exposure
            
                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 8], estimation_factor_exposure.ix[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]).fit()
            
                    # Predict the leverage exposure of the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.
            
                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]].fillna(0))
            
                    std_leverage_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure

               
                else :
                         
                    std_leverage_missing_data_imputed[stock].ix[date] = mean_of_past_10_days
                
  
        ###### liquidity exposure imputation ######
    
        if len(nan_combined_list[9]) != 0 :
        
            print('numbers of liquidity factor exposure missing', len(nan_combined_list[9]))

            today = std_liquidity.index.get_loc(date)
            eleven_days_before = std_liquidity.index.get_loc(date) - 10
        
    
            for stock in liquidity_nan_stock_list :
            
                # For the past 10 trading days, if not all values of factor exposure are NANs, then the factor exposure is estimated as the mean of them.
             
                mean_of_past_10_days = std_liquidity[stock].ix[eleven_days_before:today].mean()
            
                if np.isnan(mean_of_past_10_days) :
                
                    # obtain the industry index that the stock belongs to
           
                    industry_index = stocks_industry_classification_on_current_day[stock]
            
                    # A list of stocks that belong to the same industry
            
                    same_industry_stock_list = stocks_industry_classification_on_current_day[stocks_industry_classification_on_current_day == industry_index].index
            
                    # A list of stocks that belong to the same industry as well as no factor expsoure missing.
            
                    estimation_stock_list = [x for x in same_industry_stock_list if x not in factors_exposure_missing_stock_list]
            
                    # Subset the dataframe of factor exposure for missing data estimation
            
                    estimation_factor_exposure = factor_exposure.ix[estimation_stock_list].copy()
            
                    # Regress the liquidity exposure against all the other factors' exposure
            
                    results_ols = sm.OLS(estimation_factor_exposure.ix[:, 9], estimation_factor_exposure.ix[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]).fit()
            
                    # Predict the liquidity exposure for the stock
            
                    # fillna to replace NAN with 0, in this case the factor exposure with NAN is ingored in prediction.

                    predicted_missing_factor_exposure = results_ols.predict(factor_exposure.ix[stock, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]].fillna(0))
            
                    std_liquidity_missing_data_imputed[stock].ix[date] = predicted_missing_factor_exposure

        
                else :
                         
                    std_liquidity_missing_data_imputed[stock].ix[date] = mean_of_past_10_days
                

          
            
    # benchmark beta
    
    complete_path = os.path.join(temp_path, "std_benchmark_beta_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_benchmark_beta_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_benchmark_beta_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_benchmark_beta_missing_data_imputed[2:], output)

    output.close()


    # momentum
    
    complete_path = os.path.join(temp_path, "std_momentum_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_momentum_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_momentum_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_momentum_missing_data_imputed[2:], output)

    output.close()


    # reversal
    
    complete_path = os.path.join(temp_path, "std_reversal_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_reversal_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_reversal_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_reversal_missing_data_imputed[2:], output)

    output.close()


    # size
    
    complete_path = os.path.join(temp_path, "std_size_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_size_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_size_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_size_missing_data_imputed[2:], output)

    output.close()


    # earning yield
    
    complete_path = os.path.join(temp_path, "std_earning_yield_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_earning_yield_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_earning_yield_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_earning_yield_missing_data_imputed[2:], output)

    output.close()


    # volatility
    
    complete_path = os.path.join(temp_path, "std_volatility_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_volatility_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_volatility_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_volatility_missing_data_imputed[2:], output)

    output.close()
            
            
    # growth
    
    complete_path = os.path.join(temp_path, "std_growth_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_growth_missing_data_imputed, output)

    output.close()
    
    complete_path = os.path.join(results_path, "std_growth_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_growth_missing_data_imputed[2:], output)

    output.close()


    # value
    
    complete_path = os.path.join(temp_path, "std_value_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_value_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_value_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_value_missing_data_imputed[2:], output)

    output.close()


    # leverage
    
    complete_path = os.path.join(temp_path, "std_leverage_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_leverage_missing_data_imputed, output)

    output.close()
    
    complete_path = os.path.join(results_path, "std_leverage_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_leverage_missing_data_imputed[2:], output)

    output.close()


    # liquidity
    
    complete_path = os.path.join(temp_path, "std_liquidity_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_liquidity_missing_data_imputed, output)

    output.close()

    complete_path = os.path.join(results_path, "std_liquidity_missing_data_imputed.pkl")

    output = open(complete_path,'wb')

    pickle.dump(std_liquidity_missing_data_imputed[2:], output)

    output.close()
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    
    print('style factor exposure imputation is done')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')








