'''
使用了merge_grid_meo的合并卫星的气象数据
合并beijing_17_18_aq质量站数据

'''
#(step4)将merge_18_grid_meo_station.csv文件与air_2018_3_31.csv与气象站融合的空气质量与气象数据文件进行链接#
#将每个空气监测站点的气象数据与空气质量数据融合， 得到每个空气监测站的气象数据与空气质量数据
import pandas as pd
from math import*
import matplotlib.pyplot as plt
from process.data_merge_process.data_process.tools import *

def merge_agm():

                #print(df_aq_grid_drop.loc[index].values[0])
    # 根据经纬度聚类质量观测站 和 气象观测站

    # 读入数据文件
    aq_data = pd.read_csv("/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/origion_data/air_2018_3_31.csv")
    meo_data = pd.read_csv("/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/merge_grid_meo.csv")

    # 开始合并流程
    print("==================开始合并流程===============")
    print("==================处理气象数据、增加对应的质量站列=========开始======")
    df_meo = pd.DataFrame(meo_data,index=None,columns=['stationName','utc_time','temperature','pressure','humidity','wind_direction','wind_speed/kph'])
    df_meo.columns = ['stationId_aq','utc_time','temperature','pressure','humidity','wind_direction','wind_speed/kph']
    print(df_meo.head())


    print("==================处理气象数据、增加对应的质量站列=========结束======")
    print(df_meo.tail(100))

    print("==================处理质量数据集、删减到需要合并的数据集===开始============")
    df_aq = pd.DataFrame(aq_data,index=None,columns=['station_id','time','NO2_Concentration','CO_Concentration','SO2_Concentration','PM25_Concentration','PM10_Concentration','O3_Concentration'])
    df_aq.columns = ['stationId_aq','utc_time','NO2','CO','SO2','PM2.5','PM10','O3']

    df_aq = df_aq.drop_duplicates(['stationId_aq','utc_time'])
    print(df_aq.head())


    print("==================开始处理质量数据集、删减到需要合并的数据集===结束===============")
    print("==================合并的数据集========开始=======")
    df_merge = pd.merge(df_meo,df_aq,on=['stationId_aq','utc_time'],how='right')
    print(df_aq.head())
    #df_merge = pd.DataFrame(df_merge)
    #print(df_merge.head())
    #df_merge.to_csv('/home/fly/PycharmProjects/DeepST-KDD/for_submit_data/final_merge_aq_grid_meo2.csv',index=False)
    print("==================合并的数据集========结束=======")
    print('天气数据总记录数===》',df_aq.shape[0])
    print('合并后总记录数===》',df_merge.shape[0])


    # print("==================处理插值========开始=======")
    # # 全字段插值方法
    # def interpolation_all_columns():
    #     pass
    #
    # print("==================处理插值========结束=======")
    print("==================合并天气========开始=======")
    meo_weather_data = pd.read_csv("/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/merge_aq_meo.csv")
    df_meo_weather = pd.DataFrame(meo_weather_data,index=None,columns=['stationId_aq','utc_time','weather'])
    df_merge = pd.merge(df_merge,df_meo_weather,on=['stationId_aq','utc_time'],how='outer')
    #df_merge=pd.concat([df_merge,df_meo_weather],keys=['stationId_aq','utc_time'],axis=1,how='outer')
    print("==================合并天气========结束=======")
    print("==================写入文件merge_aq_meo.csv========开始=======")
    df_merge.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/final_merge_aq_grid_meo2.csv',index=False)
    print("==================写入文件merge_aq_meo.csv========结束=======")

merge_agm()