
###### 中间数据生成 ######

### 模块说明 ###

# 本模块对因子暴露度和因子收益估计需要的一些中间数据进行计算。


###### 输出数据 ######

# 1 日换手率；

# 2 基准组合（沪深300）相对于无风险利率的超额收益率；

# 3 个股相对于无风险利率的超额收益率。


### import external packages ###

import numpy as np
import pandas as pd

import datetime
import pickle
import os.path


### paths for saving files ###

temp_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/temp/"

results_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/results/"



### function for calculating the temporary results ###

def daily_turnover_rate():
    
    ### load total volume ###
    
    complete_path = os.path.join(temp_path, "total_volume.pkl")
    
    pkfl = open(complete_path,'rb')

    total_volume = pickle.load(pkfl)

    pkfl.close()

    
    ### load circulation shares ###
    
    complete_path = os.path.join(temp_path, "circulation_shares.pkl")
    
    pkfl = open(complete_path,'rb')

    circulation_shares = pickle.load(pkfl)

    pkfl.close()


    ### daily turnover rate ###

    for stock in circulation_shares.columns :
        
        # If the total_volume or the circulation_shares is NAN, then the daily_turnover_rate is NAN.

        daily_turnover_rate = total_volume[stock] / circulation_shares[stock]

        # put the results into the dataframe
    
        if stock == circulation_shares.columns[0] :
            
            df_daily_turnover_rate = pd.DataFrame(daily_turnover_rate.values, index = circulation_shares.index)
        
            df_daily_turnover_rate.columns = [stock]
        
        else :
                
            series_daily_turnover_rate = pd.DataFrame(daily_turnover_rate.values, index = circulation_shares.index)
        
            series_daily_turnover_rate.columns = [stock]
        
            # The parameter 'join' is set to be 'outer', which create an union of listed stocks of all concerning trading days.
                
            df_daily_turnover_rate = pd.concat([df_daily_turnover_rate, series_daily_turnover_rate], axis=1, join='outer')


    # If the circlation shares is 0 (why it happens?), then the turnover rate is infinite. So replace inf with NAN.

    df_daily_turnover_rate = df_daily_turnover_rate.replace(np.inf, np.nan)
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "daily_turnover_rate.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_daily_turnover_rate, output)

    output.close()
    
    # print a message 
    
    print('daily turnover rate calculation is done')




def benchmark_and_stock_daily_excess_return_rate():
    
    ### load benchmark close price ###
    
    complete_path = os.path.join(temp_path, "benchmark_close_price.pkl")
    
    pkfl = open(complete_path,'rb')

    benchmark_close_price = pickle.load(pkfl)

    pkfl.close()

    
    ### load stock close price ###
    
    complete_path = os.path.join(temp_path, "close_price.pkl")
    
    pkfl = open(complete_path,'rb')

    stock_close_price = pickle.load(pkfl)

    pkfl.close()

    
    ### load risk-free return rate ###
    
    complete_path = os.path.join(temp_path, "risk_free_return_rate.pkl")
    
    pkfl = open(complete_path,'rb')

    risk_free_return_rate = pickle.load(pkfl)

    pkfl.close()
    
    
    ### benchmark excess return rate ###

    # compute the arithmetic return

    benchmark_daily_return_rate = benchmark_close_price.pct_change().dropna(axis=0, how='all')

    # remove the "%H-%M-%S" in the time-index

    # benchmark_daily_return_rate.index = [datetime.datetime.strptime(i.strftime("%Y-%m-%d"),"%Y-%m-%d").date() for i in benchmark_daily_return_rate.index]

    benchmark_daily_excess_return_rate = [ ]

    for date in benchmark_daily_return_rate.index :
        
        # Covert the annualized risk-free return rate to daily risk-free return rate.

        # Here it is hypothesized that there are 240 trading days per year.
        
        daily_risk_free_return_rate = (((1 + risk_free_return_rate['0S'].ix[str(date)]) ** (1/240)) - 1)
    
        benchmark_daily_excess_return_rate.append(benchmark_daily_return_rate.ix[date] - daily_risk_free_return_rate)

    df_benchmark_daily_excess_return_rate = pd.Series(benchmark_daily_excess_return_rate, index = benchmark_daily_return_rate.index)
    
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "benchmark_daily_excess_return_rate.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_benchmark_daily_excess_return_rate, output)

    output.close()


    
    ### stock excess return rate ###
    
    # compute the arithmetic return

    stock_daily_return_rate = stock_close_price.pct_change().dropna(axis=0, how='all')

    stock_daily_excess_return_rate = [ ]

    for date in stock_daily_return_rate.index :

        # Covert the annualized risk-free return rate to daily risk-free return rate.

        # Here it is hypothesized that there are 240 trading days per year.
       
        daily_risk_free_return_rate = (((1 + risk_free_return_rate['0S'].ix[date]) ** (1/240)) - 1)
    
        stock_daily_excess_return_rate.append(stock_daily_return_rate.ix[date] - daily_risk_free_return_rate)

    df_stock_daily_excess_return_rate = pd.DataFrame(stock_daily_excess_return_rate, index = stock_daily_return_rate.index, columns = stock_daily_return_rate.columns)

    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "stock_daily_excess_return_rate.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_stock_daily_excess_return_rate, output)

    output.close()
    
    # print a message 
    
    print('benchmark and stock daily excess return rate calculation is done')




def intermediate_data_generation():
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')   
    print('intermediate data generation begins')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')   
 
    daily_turnover_rate()
  
    benchmark_and_stock_daily_excess_return_rate()

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')       
    print('intermediate data generation is done')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')   
    











