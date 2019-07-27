#final_merge_aq_grid_meo_with_weather.csv文件用于存放处理过天气文本的数据，如，晴天表示为1
import pandas as pd

data = pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/final_merge_aq_grid_meo_deal_new.csv')
print(data.info())
# data = data.loc[data['utc_time'] >= '2017-1-30 16:00:00']
# data = data.dropna()
#去除所有weather为空的数据，～表示去除的一种方法，因为连续20天没有数据
#data = data[~data['weather'].isnull()]
data.loc[data["weather"] == "Sunny/clear", "weather"] = 1
data.loc[data["weather"] == "Haze", "weather"] = 2
data.loc[data["weather"] == "Snow", "weather"] = 3
data.loc[data["weather"] == "Fog", "weather"] = 4
data.loc[data["weather"] == "Rain", "weather"] = 5
data.loc[data["weather"] == "Dust", "weather"] = 6
data.loc[data["weather"] == "Sand", "weather"] = 7
data.loc[data["weather"] == "Sleet", "weather"] = 8
data.loc[data["weather"] == "Rain/Snow with Hail", "weather"] = 9
data.loc[data["weather"] == "Rain with Hail", "weather"] = 10
data.loc[data["weather"] == "Hail", "weather"] = 11
data.loc[data["weather"] == "Cloudy", "weather"] = 12
data.loc[data["weather"] == "Light Rain", "weather"] = 13
data.loc[data["weather"] == "Thundershower", "weather"] = 14
data.loc[data["weather"] == "Overcast", "weather"] = 15

data.loc[data['weather'].isnull(),"weather"]=1
print (data.head())
print(data.weather)
data = data.interpolate()
data = data[['stationId_aq', 'utc_time', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3']]
data.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/final_merge_aq_grid_meo_deal_weather.csv')
print(data.info())
data1 = pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/final_merge_aq_grid_meo_deal_weather.csv')
data_sort = pd.DataFrame(data1.sort_values(by = ['utc_time', 'stationId_aq']))
#data_sort.to_csv('/home/fly/PycharmProjects/gmh/18_data_merge_process/data/sort.csv', index=False)
#data2=pd.read_csv('/home/fly/PycharmProjects/gmh/18_data_merge_process/data/sort.csv')
data_sort = data_sort[['utc_time','stationId_aq','temperature','pressure','humidity','wind_direction','wind_speed/kph','weather','NO2','CO','SO2','PM2.5','PM10','O3']]
data_sort.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/final_merge_aq_grid_meo_deal_weather.csv', index=False)
data_sort.corr().to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/feature_important_show.csv')
# df_data_sort=pd.DataFrame(data_sort)
# print(df_data_sort.head())
