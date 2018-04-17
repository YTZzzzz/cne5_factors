

###### 数据读取 ######

### 模块说明 ###

# 本模块从 RQData 读取计算所需的数据，并保存在本地。


###### 工程备忘 ######

### 1 get_price 的各个变量，以及 circulation_shares 变量，需额外读取前一年的历史数据，用于之后的贝塔因子、长期动量/波动率/流动性等细分因子的暴露度计算；

### 2 其余变量需额外读取前 17 个交易日的数据，用于后期的细分因子权重计算，以及缺失值填补。





### import RQData ###
    
import rqdatac
from rqdatac import *
rqdatac.init('jjj', '123')
    
    
### import external packages ###

import numpy as np    
import pandas as pd
import datetime
import pickle
import os.path


### paths for saving files ###

temp_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/temp/"

results_path = "/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/data/results/"


### function for obtaining data from RQData ###

def listed_stocks(stocks, time_period, time_period_with_past_year):
    
    # Create a dataframe with 'True'
    
    df_listed_stocks = pd.DataFrame(index = time_period, columns = stocks)

    df_listed_stocks = df_listed_stocks.fillna('True')

    
    # Loop through all selected trading days to check whether the stocks were listed as well as not delisted.

    for date in time_period :
        
        #print(date)
        
        date_str = str(date)   
                    
        listed_stocks = [i for i in instruments(list(all_instruments(type='CS').order_book_id)) if i.listed_date <= date_str and \
                        (i.de_listed_date == '0000-00-00' or date_str < i.de_listed_date)]
                           
        # Extract the order_book_ids
                    
        listed_stocks_ids = [ ]

        for j in range(0, len(listed_stocks)) :
            
            listed_stocks_ids.append(listed_stocks[j].order_book_id)
    
        
        #print(date, len(listed_stocks_ids))
                        
                        
        # Fill the spots in dataframe with 'False' if stocks were not listed or it was delisted.
    
        for k in range(0, len(stocks)):
            
            if stocks[k] not in listed_stocks_ids :
                
                df_listed_stocks[stocks[k]].ix[date] = 'False'
   
   
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_listed_stocks, output)

    output.close()

    
    # Obtain the listed stocks of extra 21 trading days for short-term volatility exposure estimation

    # Create a dataframe with 'True'
    
    time_period_with_21_extra_days = time_period_with_past_year[time_period_with_past_year.index(time_period[0]) - 21:]

    df_listed_stocks_with_21_extra_trading_days = pd.DataFrame(index = time_period_with_21_extra_days, columns = stocks)

    df_listed_stocks_with_21_extra_trading_days = df_listed_stocks_with_21_extra_trading_days.fillna('True')
    

    # Loop through all selected trading days to check whether the stocks were listed as well as not delisted.

    for date in time_period_with_21_extra_days :
        
        #print(date)
        
        date_str = str(date)   
                    
        listed_stocks = [i for i in instruments(list(all_instruments(type='CS').order_book_id)) if i.listed_date <= date_str and \
                        (i.de_listed_date == '0000-00-00' or date_str < i.de_listed_date)]
                           
        # Extract the order_book_ids
                    
        listed_stocks_ids = [ ]

        for j in range(0, len(listed_stocks)) :
            
            listed_stocks_ids.append(listed_stocks[j].order_book_id)
    
        
        #print(date, len(listed_stocks_ids))
                        
                        
         # Fill the spots in dataframe with 'False' if stocks were not listed or it was delisted.
    
        for k in range(0, len(stocks)):
            
            if stocks[k] not in listed_stocks_ids :
                
                df_listed_stocks_with_21_extra_trading_days[stocks[k]].ix[date] = 'False'
    
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks_with_21_extra_trading_days.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_listed_stocks_with_21_extra_trading_days, output)

    output.close()



    # Create an empty dataframe filled with 'True'
        
    df_listed_stocks_for_133_trading_days = pd.DataFrame(index = time_period, columns = stocks)

    df_listed_stocks_for_133_trading_days = df_listed_stocks_for_133_trading_days.fillna('True')

    
    # Loop through all selected trading days to check whether the stocks were listed for 133 trading days as well as not delisted.

    for date in time_period :
        
        date_str = str(date)   
        
        one_hundred_and_thirty_three_days_before = str(time_period_with_past_year[time_period_with_past_year.index(date) - 133])
 
    
        listed_stocks = [i for i in instruments(list(all_instruments(type='CS').order_book_id)) if i.listed_date <= one_hundred_and_thirty_three_days_before and \
                        (i.de_listed_date == '0000-00-00' or date_str < i.de_listed_date)]
                        
   
        # Extract the order_book_ids
                    
        listed_stocks_ids = [ ]

        for j in range(0, len(listed_stocks)) :
            
            listed_stocks_ids.append(listed_stocks[j].order_book_id)
         
        #print(date, len(listed_stocks_ids))
                        
                        
        # Fill the spots in dataframe with 'False' if stocks were not listed for 133 trading days or it was delisted.
    
        for k in range(0, len(stocks)):
            
            if stocks[k] not in listed_stocks_ids :
                
                df_listed_stocks_for_133_trading_days[stocks[k]].ix[date] = 'False'
        
    ### ouput the results ###
        
    # to the temp directory 
        
    complete_path = os.path.join(temp_path, "df_listed_stocks_for_133_trading_days.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_listed_stocks_for_133_trading_days, output)

    output.close()

        
    # to the result directory 
        
    complete_path = os.path.join(results_path, "df_listed_stocks_for_133_trading_days.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_listed_stocks_for_133_trading_days.ix[16:], output)

    output.close()
    
    # print a message 
    
    print('listed stocks loading is done')


    
def st_and_under_trading_stocks(stocks, time_period, time_period_with_past_year):
    
    ###### st stocks ######

    df_st_stocks = is_st_stock(stocks.tolist(), start_date = time_period[0], end_date = time_period[-1])
    
    df_st_stocks = df_st_stocks.astype(str)
  
  
    ### ouput the results ###
      
    complete_path = os.path.join(temp_path, "st_stocks.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_st_stocks, output)

    output.close()

    
    ### under trading stocks ###

    # start from 5 trading days before, so that we can determine whether it was suspended for the last 5 trading days.

    five_days_before = str(time_period_with_past_year[time_period_with_past_year.index(time_period[0]) - 5])

    # stock was under trading if its total volume was larger than 0
 
    under_trading_stocks = get_price(list(stocks),start_date = five_days_before, end_date = time_period[-1], fields='TotalVolumeTraded') > 0
    
    under_trading_stocks = under_trading_stocks.astype(str)


    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "under_trading_stocks.pkl")

    output = open(complete_path,'wb')

    pickle.dump(under_trading_stocks, output)

    output.close()
    
    # print a message 
    
    print('st and under-trading stocks loading is done')




def price_and_volume_data(stocks, a_year_before, enddate):
    
    ### close price ###

    close_price = get_price(list(stocks), start_date = a_year_before, end_date = enddate, frequency='1d', fields = 'ClosingPx', adjust_type='none', country='cn')

    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "close_price.pkl")

    output = open(complete_path,'wb')

    pickle.dump(close_price, output)

    output.close()
    

    ### total volume traded ###

    Total_Volume_Traded = get_price(list(stocks), start_date = a_year_before, end_date = enddate, frequency='1d', fields = 'TotalVolumeTraded', adjust_type='none', country='cn')

    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "total_volume.pkl")

    output = open(complete_path,'wb')

    pickle.dump(Total_Volume_Traded, output)

    output.close()


    ### benckmark close price ###

    benchmark_close_price = get_price('CSI300.INDX', start_date = a_year_before, end_date = enddate,fields='ClosingPx')
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "benchmark_close_price.pkl")

    output = open(complete_path,'wb')

    pickle.dump(benchmark_close_price, output)

    output.close()


    ### risk free return rate ###

    risk_free_return_rate = get_yield_curve(start_date = a_year_before, end_date = enddate, tenor=None, country='cn')
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "risk_free_return_rate.pkl")

    output = open(complete_path,'wb')

    pickle.dump(risk_free_return_rate, output)

    output.close()
    
    # print a message 
    
    print('price & volume data loading is done')




def fundamental_data(stocks, enddate, time_period, a_year_before):
    
    ### circulation shares ###

    circulation_shares = get_shares(list(stocks), start_date = a_year_before, end_date = enddate, fields='circulation_a', country='cn') 
    
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "circulation_shares.pkl")

    output = open(complete_path,'wb')

    pickle.dump(circulation_shares, output)

    output.close()
    
    # we need extra 5 trading days for atomic descriptor weight estimation.

    df_fundamental = get_fundamentals(query(fundamentals.eod_derivative_indicator.pe_ratio,\
                                            fundamentals.financial_indicator.operating_cash_flow_per_share,\
                                            fundamentals.financial_indicator.capital_reserve_per_share,\
                                
                                            fundamentals.eod_derivative_indicator.market_cap,\
                               
                                            fundamentals.financial_indicator.inc_revenue,\
                                            fundamentals.financial_indicator.inc_total_asset,\
                                            fundamentals.financial_indicator.inc_operating_revenue,\
                                            fundamentals.financial_indicator.inc_gross_profit,\
                                
                                            fundamentals.balance_sheet.total_equity,\
                                                           
                                            fundamentals.balance_sheet.total_liabilities,\
                                            fundamentals.balance_sheet.total_assets,\
                                            ),enddate, str(len(time_period))+'d')
                                            
    
    # replace NAN in pe_ratio with zeros

    df_fundamental['pe_ratio'] = df_fundamental['pe_ratio'].replace(np.nan, 0)
                                        
    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "df_fundamental.pkl")

    output = open(complete_path,'wb')

    pickle.dump(df_fundamental, output)

    output.close()
    
    # print a message 
    
    print('fundamental data loading is done')



def industry_label(time_period):
    
        
    ### load listed stocks ###
    
    complete_path = os.path.join(temp_path, "df_listed_stocks.pkl")
    
    pkfl = open(complete_path,'rb')

    listed_stocks = pickle.load(pkfl)

    pkfl.close()

    
    industry_label_list = ['801010.INDX', '801020.INDX', '801030.INDX', '801040.INDX', '801050.INDX', '801080.INDX', '801110.INDX',\
                           '801120.INDX', '801130.INDX', '801140.INDX', '801150.INDX', '801160.INDX', '801170.INDX', '801180.INDX',\
                           '801200.INDX', '801210.INDX', '801230.INDX', '801710.INDX', '801720.INDX', '801730.INDX', '801740.INDX',\
                           '801750.INDX', '801760.INDX', '801770.INDX', '801780.INDX', '801790.INDX', '801880.INDX', '801890.INDX']
               
    
    # Create an dataframe filled with zeros

    stocks_industry_classification = pd.DataFrame('no_label', index = listed_stocks.index, columns = listed_stocks.columns)


    for date in time_period :
                
        for label in industry_label_list:
            
            industry_label = shenwan_industry(label, date = str(date))
            
            stocks_industry_classification = stocks_industry_classification.set_value(date, industry_label, label)


    ### ouput the results ###
    
    complete_path = os.path.join(temp_path, "stocks_industry_classification.pkl")

    output = open(complete_path,'wb')

    pickle.dump(stocks_industry_classification, output)

    output.close()
    
    # print a message 
    
    print('industry label loading is done')



def data_acquisition_from_rqdata(begindate,  enddate):
   
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
    print('data acquisition begins')
    print('~~~~~~~~~~~~~~~~~~~~~~~~')
    
    
    ###### Initial setting ######
    
    # Obtain the stocks of the whole universe

    stocks = all_instruments(country='cn', type='CS')['order_book_id'].values
   
   
    # Obtain the trading days between the begindate and enddate
   
    original_time_period = get_trading_dates(begindate, enddate)
    
    # Stop the execution if there IS no trading day in the selected time period.
    
    error_flag = 0
    
    if len(original_time_period) == 0 :
        
        print('ERROR: there is no trading day in the selected time period !')
        
        error_flag = 1
        
        return error_flag
        
        
    else :

        # Covert the str format to datetime format
    
        begindate_datetime = datetime.datetime.strptime(begindate, "%Y-%m-%d").date()
    
    
        # we need extra 17 trading days for atomic descriptor weight estimation and missing data imputation.
    
        a_hundred_days_before = begindate_datetime - datetime.timedelta(days=100)
    
        time_period = get_trading_dates(a_hundred_days_before, enddate)
    
        time_period = time_period[-(len(original_time_period) + 17):]
    
    
        # We need extra year for price & volume factor exposure estimation
    
        a_year_before = begindate_datetime - datetime.timedelta(days=365)
    
        time_period_with_past_year = list(get_trading_dates(a_year_before, enddate))
    
    
        ###### data acquisition ######
    
        listed_stocks(stocks, time_period, time_period_with_past_year)
        
        st_and_under_trading_stocks(stocks, time_period, time_period_with_past_year)
        
        price_and_volume_data(stocks, a_year_before, enddate)
    
        fundamental_data(stocks, enddate, time_period, a_year_before)
    
        industry_label(time_period)

        print('~~~~~~~~~~~~~~~~~~~~~~~~')    
        print('data acquisition is done')
        print('~~~~~~~~~~~~~~~~~~~~~~~~')
    
        return error_flag














