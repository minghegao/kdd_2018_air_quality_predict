import pandas as pd

def deal_weather_pred():
    ######################  1.加载天气预报并排序  ###################
    beijing_weather=pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/origion_data/weather_pred_data/Beijing_weather.csv',parse_dates=['datetime'])
    new_beijing_weather=beijing_weather.sort_values(by=['datetime','station_id'])

    ######################  2.对数据进行整理，规范做实验所需的时间范围  ###################
    start_date=pd.to_datetime('2018-03-31 07:00:00')
    end_date = pd.to_datetime('2018-05-29 06:00:00')
    beijing_weather=new_beijing_weather[(start_date<=new_beijing_weather['datetime'])&(new_beijing_weather['datetime']<=end_date)]
    #beijing_weather.to_csv('./for_look_weather.csv', index=False)


    ######################  3.该做一些数据的预处理  ###################
    #去重
    beijing_weather.drop_duplicates(['station_id','datetime'],keep='first')
    beijing_weather.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/weather_prediction.csv', index=False)


    #检查是否缺少时刻点
    print(beijing_weather['datetime'].shape)
if __name__=='__main__':
    deal_weather_pred()