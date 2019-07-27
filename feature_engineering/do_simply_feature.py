import pandas as pd
import numpy as np
pd.set_option('display.max_columns',5000)

air_data=pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/fill_5_hour_data.csv')
weather_pred_data=pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/weather_prediction.csv',parse_dates=['datetime'])

def do_ori_air_feature():
    '''
    此函数用于进行空气质量原始特征的一些删减和变换    暂时不变幻了，之后会考虑使用自动化造特征工具 ，或者模拟退火等方法造一些特征，   留好接口，在此接口上设计特征
    '''
    #作简单的去重
    air__data=air_data.drop_duplicates(['stationId_aq','utc_time'])
    print('输出去重之后的形状：',air__data.shape)
    air__data.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/feature_eng_data/air_data.csv')

def do_ori_weather_pred_data():
    '''
    函数用于对于天气预报作特征变换
    '''

    #weather_pred_data.drop(labels='datetime', axis=1, inplace=True)
    weather_pred_data.drop(labels='lat', axis=1, inplace=True)
    weather_pred_data.drop(labels='long', axis=1, inplace=True)

    # ToDo: Undo. Removed because they have a lot of empty cells. Replace with more meaningful values
    weather_pred_data.drop(labels='ozone', axis=1, inplace=True)
    weather_pred_data.drop(labels='precipIntensity', axis=1, inplace=True)
    weather_pred_data.drop(labels='precipProbability', axis=1, inplace=True)
    weather_pred_data.drop(labels='pressure', axis=1, inplace=True)
    weather_pred_data.drop(labels='uvIndex', axis=1, inplace=True)
    weather_pred_data.drop(labels='windGust', axis=1, inplace=True)
    weather_pred_data.drop(labels='cloudCover', axis=1, inplace=True)
    weather_pred_data.drop(labels='precipType', axis=1, inplace=True)
    weather_pred_data.drop(labels='visibility', axis=1, inplace=True)

    weather_pred_data.drop(labels='cloudCoverError', axis=1, inplace=True)
    weather_pred_data.drop(labels='pressureError', axis=1, inplace=True)
    weather_pred_data.drop(labels='windSpeedError', axis=1, inplace=True)
    weather_pred_data.drop(labels='temperatureError', axis=1, inplace=True)
    weather_pred_data.drop(labels='windBearingError', axis=1, inplace=True)
    weather_pred_data.drop(labels='precipAccumulation', axis=1, inplace=True)
    weather_pred_data.drop(labels='time', axis=1, inplace=True)
    # Make categorical columns for for time based attributes
    weather_pred_data['hour'] = weather_pred_data['datetime'].dt.hour
    weather_pred_data['day'] = weather_pred_data['datetime'].dt.day
    weather_pred_data['month'] = weather_pred_data['datetime'].dt.month
    weather_pred_data['dayofweek'] = weather_pred_data['datetime'].dt.dayofweek


    #使用one-hot编码，分别对summary 天气和  station_id 站点进行独热展开
    #we=pd.get_dummies(weather_pred_data,columns=['summary'])
    #weather_data=pd.get_dummies(we,columns=['station_id'])

    # 下面的方式会自动对文本类型数据进行毒热编码，    可以看出上面的程序是有指定列上数值one-hot编码。
    weather_pred_data['summary'] = weather_pred_data['summary'].astype('category')
    weather_pred_data['icon'] = weather_pred_data['icon'].astype('category')

    weather_data = pd.get_dummies(weather_pred_data, columns=['station_id'])
    weather_data = pd.get_dummies(weather_data, prefix='dum', drop_first=True)


    print('::::',weather_data.head(20))
    weather_data.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/feature_eng_data/weather_pred.csv')


#do_ori_weather_pred_data()