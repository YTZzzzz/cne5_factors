


###### RQBeta 数据自动化更新 ######


### 模块说明 ###

# RQBeta 数据自动化更新的主函数。


### 运行环境 ###

# Mac OS X EI Capitan 版本 10.11.6

# Anaconda 平台

# IDE：Spyder-2.3.9


### 代码运行/数据自动更新步骤 ###

### 1 初始设定

# （1） 在本模块中，设定 sys.path.append(...) 路径（RQBeta 源代码所在路径）；

# （2） 在 RQBeta 的各子模块文件开头部分，设定临时文件保存路径（temp_path）和最终计算结果保存路径（results_path）；

# （3） 在 preprocessing/data_acquisition 模块文件中，设定 RQData 的用户账号和密码；


### 2 数据更新

# （1） 在 main 函数中设定起止日期（起止日期可以相同，更新一个交易日的数据）；

# （2） 编译本模块所有代码；

# （3） 输入 main() 指令，程序将打印出一系列信息，提示代码运行进度。


### 计算结果 ###

# 在程序运行完毕后，results_path 的文件夹中将保存如下计算结果:

# 1 两个用于数据缺失值检查的 dataframes (stock_nan_check 和 variable_nan_check);

# 2 一个标记上市达到 133 个交易日的股票的 dataframe（在特定交易日，上市达到 133 个交易日的股票标记为 True，否则为 False）;

# 3 十个已作缺失值处理的 RQ 风格因子暴露度 dataframes;

# 4 一个 RQ 风格因子市场中性化收益 dataframe。



###### 工程备忘 ######

# 1 在计算过程中，由于涉及到的原始数据和中间数据种类较多，为简化代码，避免多个数组在各个模块中传递而造成的模块间复杂依赖关系，
# 在目前 RQBeta 的实现方案中，原始数据，及计算过程中得到的中间数据，均保存在本地的临时文件夹中。在之后的计算中，各模块按照需要，再在临时文件夹中读取相关数据。


# 2 如果选择的起止日期之间不包含任何的交易日，则程序会打印错误信息，并停止计算。目前 main() 函数中的 sys.exit() 未能实现自动退出程序，尚需进一步测试。


### path for searching the RQBeta source code ###

import sys

sys.path.append("/Users/jjj728/Dropbox/quant_trading/RQBeta/automated_scripts/")


### load the RQBeta packages ###

from preprocessing.data_acquisition import data_acquisition_from_rqdata
from preprocessing.data_inspection import data_inspection
from preprocessing.intermediate_data_generation import intermediate_data_generation
from preprocessing.estimation_universe import estimation_universe_construction


from factor_exposure.atomic_descriptors import atomic_descriptors
from factor_exposure.style_factor_exposure import style_factor_exposure
from factor_exposure.style_factor_exposure_imputation import style_factor_exposure_imputation


from factor_return.market_neutralized_factor_return import factor_return_estimation



###### RQBeta automated data update ######

def main():
    
    ### time period for factor exposure and factor return estimation ###
    
    begindate = '2016-12-22'

    enddate = '2016-12-24'
    
    
    ###### preprocessing ######
    
    error_flag = data_acquisition_from_rqdata(begindate, enddate)
    
    # quit the execution if there is no trading day contained in the selected time period.
    
    if error_flag == 1:
        
        sys.exit()
    
    data_inspection()
    
    intermediate_data_generation()

    estimation_universe_construction()


    ###### style factor exposure ######
    
    atomic_descriptors()
    
    style_factor_exposure()
    
    style_factor_exposure_imputation()
    
   
    ###### factor return ######
    
    factor_return_estimation()
    


# main() function can only be executed directly, and it won't be executed if it is imported in another module.


if __name__ == '__main__':
    main()