

###### 股票池构建 ######

### 模块说明 ###

# 本模块对市场中收益可能存在异常、或噪声较大的股票进行剔除，以构建一个能代表市场普遍情况的，适用于参数估计，及各类测试的股票池。

# 具体而言，我们剔除符合以下条件的个股：

#（1）ST股；

#（2）上市少于 133 个交易日（约半年）；

#（3）前 5 个交易日曾停牌；

#（4）前 5 个交易日的日收益曾出现异常，即日收益小于等于-10%，或大于等于 10%；

#（5）前 22 个交易日的平均日换手率处于市场的后10%。


###### 输出数据 ######

# 经筛选后的测试股票池；


### import external packages ###

import numpy as np
import pandas as pd

import datetime
import pickle
import os.path


### paths for saving files ###

temp_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/temp/"

results_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/results/"



def under_trading_for_the_past_five_trading_days():
    
    ### under trading stocks ###
    
    complete_path = os.path.join(temp_path, "under_trading_stocks.pkl")
    
    pkfl = open(complete_path,'rb')

    under_trading_stocks = pickle.load(pkfl)

    pkfl.close()
    
    # stocks that were under trading are denoted as 1, otherwise 0.
    
    # Covert the label of stocks that were under trading as 1, otherwise 0.
    
    under_trading_stocks = under_trading_stocks.replace('True', 1)
    under_trading_stocks = under_trading_stocks.replace('False', 0)
    
    # if they were under trading for the last 5 trading days, the sum wil be equal to 5.

    stocks_under_trading_for_the_past_5_days = under_trading_stocks.rolling(window=5).sum()

    stocks_under_trading_for_the_past_5_days = stocks_under_trading_for_the_past_5_days.dropna()

    # stocks that were under trading are denoted as True, otherwise denoted as False.

    stocks_under_trading_for_the_past_5_days = stocks_under_trading_for_the_past_5_days.replace([0.0, 1.0, 2.0, 3.0, 4.0], 'False')

    stocks_under_trading_for_the_past_5_days = stocks_under_trading_for_the_past_5_days.replace([5.0], 'True')
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "stocks_under_trading_for_the_past_5_days.pkl")

    output = open(complete_path,'wb')

    pickle.dump(stocks_under_trading_for_the_past_5_days, output)

    output.close()
    
    print('Select stocks under trading for the past 5 trading days is done.')



def turnover_rate_in_top_90_percent():
    
    ### load stocks listed for 133 trading days ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks_for_133_trading_days.pkl")
    
    pkfl = open(complete_path,'rb')

    listed_stocks = pickle.load(pkfl)

    pkfl.close()
    
    
    ### load st stocks ###
    
    complete_path = os.path.join(temp_path, "st_stocks.pkl")
    
    pkfl = open(complete_path,'rb')

    st_stocks = pickle.load(pkfl)

    pkfl.close()

    
    ### load daily turnover rates ###
    
    complete_path = os.path.join(temp_path, "daily_turnover_rate.pkl")
    
    pkfl = open(complete_path,'rb')

    daily_turnover_rate = pickle.load(pkfl)

    pkfl.close()

    
    # If not all daily turnover rates were missing for the past 22 trading day, it returns a value.

    rolling_mean_turnover_rate = daily_turnover_rate.rolling(window=22).mean()

    # Remove the frist 21 trading days with NaNs.

    rolling_mean_turnover_rate = rolling_mean_turnover_rate.iloc[21:,]


    # Replace NaNs with zeros, so that we can filter stocks with daily turnover rate being NANs or Zeros (i.e., total volume is zero) simultaneously.

    rolling_mean_turnover_rate  = rolling_mean_turnover_rate.replace(np.nan, 0)


    # Create an dataframe 

    turnover_rate_in_top_90_percent = rolling_mean_turnover_rate.copy()
    
    # Keep a slice of the dataframe of interest
    
    turnover_rate_in_top_90_percent = turnover_rate_in_top_90_percent[st_stocks.columns].ix[st_stocks.index] 
        


    for date in listed_stocks.index :
                  
        stock_list = [ ]
        
        for stock in listed_stocks.columns:
            
            # Stocks that were note listed or belonged to 'ST' stocks are filtered out.
        
            # Sstocks with daily turnover rate being NANs or Zeros are filtered out.

            if (st_stocks[stock].ix[date] == 'False') and (listed_stocks[stock].ix[date] == 'True') and (abs(rolling_mean_turnover_rate[stock].ix[str(date)]) > 0.000001) :
                
                stock_list.append(stock)
               
        rolling_mean_turnover_rate_at_a_day = rolling_mean_turnover_rate[stock_list].ix[str(date)]
    
        ten_percent_quantile = rolling_mean_turnover_rate_at_a_day.quantile(.1)
        
        # Label the stocks with turnover rate in top 90 %  as 1, otherwise 0.
    
        turnover_rate_in_top_90_percent.ix[date][turnover_rate_in_top_90_percent.ix[date] <  ten_percent_quantile] = 0

        turnover_rate_in_top_90_percent.ix[date][turnover_rate_in_top_90_percent.ix[date] >  ten_percent_quantile] = 1
        
    
    # Covert the labels to 'True' and 'False'.
    
    turnover_rate_in_top_90_percent = turnover_rate_in_top_90_percent.replace(1.0, 'True')
    turnover_rate_in_top_90_percent = turnover_rate_in_top_90_percent.replace(0.0, 'False')

    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "turnover_rate_in_top_90_percent.pkl")

    output = open(complete_path,'wb')

    pickle.dump(turnover_rate_in_top_90_percent, output)

    output.close()
    
    print('Select stocks with turnover rate in top 90 percent is done.')

    

def check_anomaly_in_daily_return_rate_for_past_5_trading_days():
    
    ### load stock close price ###
    
    complete_path = os.path.join(temp_path, "close_price.pkl")
    
    pkfl = open(complete_path,'rb')

    stock_close_price = pickle.load(pkfl)

    pkfl.close()
    
    ### stock excess return rate ###
    
    # Note that the excess return rate should not be used here.

    stock_daily_return_rate = stock_close_price.pct_change().dropna(axis=0, how='all')

    stocks_with_anomaly_in_daily_return_rate = stock_daily_return_rate.copy()

    
    # Stocks that with unreasonable values or NANs are denoted as 1.

    stocks_with_anomaly_in_daily_return_rate[abs(stocks_with_anomaly_in_daily_return_rate) >= 0.1] = 1


    # Stocks that with reasonable values are be denoted as 0.

    stocks_with_anomaly_in_daily_return_rate[abs(stocks_with_anomaly_in_daily_return_rate) < 0.1] = 0
    
    
    # with the type conversion, stocks with 0.0 or NAN are denoted as 0; stocks with 1.0 are denoted as 1.

    stocks_with_anomaly_in_daily_return_rate = (stocks_with_anomaly_in_daily_return_rate == 1.0).astype(int)

    
    # If stocks that have anomaly in daily return for the past 5 trading days, the sum wil be larger than 0

    stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days = stocks_with_anomaly_in_daily_return_rate.rolling(window=5).sum()

    stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days = stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days.dropna()


    # stocks that have anomaly in daily return for the past 5 trading days are denoted as 1.

    stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days[stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days > 0] = 1
    
    # Covert the labels to 'True' and 'False'.
    
    stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days = stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days.replace(1.0, 'True')
    stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days = stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days.replace(0.0, 'False')

    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days.pkl")

    output = open(complete_path,'wb')

    pickle.dump(stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days, output)

    output.close()
    
    print('Select stocks with no anomaly in daily return rate for the past 5 trading days is done.')



def estimation_universe():
    
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

    
    ### under trading for the past 5 trading days ###
    
    complete_path = os.path.join(temp_path, "stocks_under_trading_for_the_past_5_days.pkl")
    
    pkfl = open(complete_path,'rb')

    stocks_under_trading_for_the_past_5_days = pickle.load(pkfl)

    pkfl.close()
    
    
    ### turnover rate in top 90 percent ###
    
    complete_path = os.path.join(temp_path, "turnover_rate_in_top_90_percent.pkl")
    
    pkfl = open(complete_path,'rb')

    turnover_rate_in_top_90_percent = pickle.load(pkfl)

    pkfl.close()
    
    
    ### stocks with anomaly in daily return rate for the past 5 days ###
    
    complete_path = os.path.join(temp_path, "stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days.pkl")
    
    pkfl = open(complete_path,'rb')

    stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days = pickle.load(pkfl)

    pkfl.close()


    
    # Create an dataframe filled with zeros

    estimation_universe = pd.DataFrame(index = st_stocks.index, columns = st_stocks.columns)

    estimation_universe = estimation_universe.fillna('False')


    for date in st_stocks.index :
                
        for stock in st_stocks.columns :
            
            if (st_stocks[stock].ix[date] == "False") and (listed_stocks_for_133_trading_days[stock].ix[date] == "True") and\
               (stocks_under_trading_for_the_past_5_days[stock].ix[str(date)] == "True") and (turnover_rate_in_top_90_percent[stock].ix[date] == "True")\
               and (stocks_with_anomaly_in_daily_return_rate_for_the_past_5_days[stock].ix[str(date)] == "False") :
                   
                   estimation_universe[stock].ix[date] = "True"
        
        # Check the size of estimation universe
        
        #print(date, 'size of estimation universe', (estimation_universe.ix[date] == 'True').sum())
    
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "estimation_universe.pkl")

    output = open(complete_path,'wb')

    pickle.dump(estimation_universe, output)

    output.close()
    
    print('estimation universe construction is done.')

            
    
def estimation_universe_construction():
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('estimation universe construction begins')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  

    under_trading_for_the_past_five_trading_days()
    
    turnover_rate_in_top_90_percent()
    
    check_anomaly_in_daily_return_rate_for_past_5_trading_days()
    
    estimation_universe()
 
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')   
    print('estimation universe construction is done')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    
    
    
    
