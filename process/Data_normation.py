import numpy as np
import pandas as pd
from datetime import datetime
import h5py
import time
from sklearn.preprocessing import MinMaxScaler
'''
代码将  标准化处理好的数据，按照大程序所需要的   h5文件中的内容形式，将 数据进行了 维度组合 和拼接，  并将日期数据进行格式转换， 分别放到三个列表中，之后保存h5文件中，、
等待模型的读取。 
'''
#使用  训练使用用的数据及进行标准化和反标准化
sta_origion_data=pd.read_csv('/home/fly/PycharmProjects/DeepST-KDD for_train/for_submit_data/gao_data/final_merge_aq_grid_meo_with_weather_sort.csv')

read_csvfile=pd.read_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/final_merge_aq_grid_meo_deal_weather.csv')
# ========================      对数据读取并标准化处理下（标准化和反标准化需要一个去适配）      ===========================#
scaler = MinMaxScaler(feature_range=(-1, 1))  # 数值限定到-1到1之间 看出来了，这是在同一个scaler下，将tranform转换后的数据，使用reverse还原回来，scaler有记忆的，所以在这里没用
#fit之前先挑选出全都是数值型的列     这里针对数值型进行按序列标准化之后再进行拼接，这里在使用fit_transform之后列的标题会丢掉，我又重加了进去。。。   标准化时候用了天气，但是训练时候没用天气，但是这里天气要加
scaler.fit(sta_origion_data[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']])  # Compute the minimum and maximum to be used for later scaling 这是个fit过程， 这个过程会 计算以后用于缩放的平均值和标准差， 记忆下来
data1 = pd.DataFrame(scaler.fit_transform(read_csvfile[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]),columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3'])  # Fit to data, then transform it. 使用记忆的数据进行转换
data2 = read_csvfile[['stationId_aq', 'utc_time']]
frames = [data1, data2]  #在列上进行拼接。。。    还有很多链接方式的选择，左右、外链接等方式，全在书上，方式很多。。。
read_csvfile = pd.concat(frames, axis=1)
pred_array=pd.DataFrame(read_csvfile)
pred_array.to_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/final_merge_aq_grid_meo_deal_weather_normation.csv',index=False)


# ========================      插曲-借助进行反标准化 （这里很大的问题是没有天气信息，我随便给的天气，发现预测结果还不对，对着那）     ===========================#
# ok_filename = pd.read_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/the_predict_data.csv')
#
# pred_array = pd.DataFrame(scaler.inverse_transform(ok_filename[['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]),columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3'])  #使用记忆的数据反转换
# pred_array1 = ok_filename[['utc_time']]
# frames1 = [pred_array, pred_array1]
# read_csvfile1 = pd.concat(frames1, axis=1)
# #pred_array=pd.DataFrame(read_csvfile1,columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3','utc_time'])
# pred_array=pd.DataFrame(read_csvfile1)
# pred_array.to_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/the_predict_data_renormation.csv',index=False)
#
