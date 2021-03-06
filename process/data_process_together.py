#----------------  这里应该把所有的自动的处理过程都放这里  -----------------#
'''
把所有的处理过程都融在一起

'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#----------------------------------------    1-   将站点数据和气象数据进行合并------------------------------------------#
'''
这里调用其他文件里面的函数来是进行数据的合并
1.先用 merge_meo_grid做下整体插值，获得所有站点的气象数据的方式,能得到每个站点再当前时刻该有的气象数据结构

merge_aq_meo是不需要的，因为数据里面没有气象站数据， 第一次合并搞好了，之前整了一次数据乌龙，经纬度位置不对应出现问题了
2.再作merge_aq_gird_meo就完全搞定了，将第一步处理的结果数据保存下来。。
'''
#再那两个融合文件里，包括  dara_preparation进行时间数据调整
#----------------------------------------    2-   去异常值和线性插值        ------------------------------------------#
data=pd.read_csv("/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/final_merge_aq_grid_meo2.csv")
#data=pd.read_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/final_merge_aq_grid_meo_sort_new.csv")
print("原始:\n")
print(data.info())
#print(data.describe())
#处理温度的异常值，温度大于100为观测错误
#    先统计出均值，之后用均质进行  填充。。。   理好逻辑，初始赋0对本体是没有影响的。
#-------------------------------------------------------------------处理温度
temp = np.array(data['temperature'])
#numpy求均值需要去除空值，～表示去空，之后再赋值，求平均，否则会出现警告 invalid value encountered in greater。。。。。。。
temp=temp[~np.isnan(temp)]
temp[temp > 50] = 0
temp[temp < -30] = 0
mean =np.mean(temp)
#print('mean:\n',mean)
data.loc[data['temperature'] > 100, 'temperature'] = mean
data.loc[data['temperature'] < -30, 'temperature'] = mean
data.loc[data['temperature'].isnull(),'temperature']=mean

#print(data['temperature'])
#-------------------------------------------------------处理压力
#处理压力的异常值，温度大于2000为观测错误
#temp = data['pressure'].dropna()
temp = np.array(data['pressure'])
temp=temp[~np.isnan(temp)]
temp[temp > 2000] = 0
temp[temp < 990] = 0
mean = np.mean(temp)
print("mean:",mean)
data.loc[data['pressure'] > 2000, 'pressure'] = mean
data.loc[data['pressure'] < 990, 'pressure'] = mean
data.loc[data['pressure'].isnull(),'pressure']=mean
print(data.info())
#-------------------------------------------------------处理湿度
temp = np.array(data['humidity'])
temp=temp[~np.isnan(temp)]
temp[temp >100] = 0
temp[temp <0] = 0
mean = np.mean(temp)
data.loc[data['humidity'] > 100, 'humidity'] = mean
data.loc[data['humidity'] <0, 'humidity'] = mean
data.loc[data['humidity'].isnull(),'humidity']=mean
#------------------------------------------------------处理风向
#处理风向的异常值,风向大于360为观测错误，但是风向越有5000异常值
#temp = data['wind_direction'].dropna()
temp = np.array(data['wind_direction'])
temp=temp[~np.isnan(temp)]
temp[temp > 360] = 0
mean = np.mean(temp)
data.loc[data['wind_direction'] > 360, 'wind_direction'] = mean
data.loc[data['wind_direction'].isnull(),'wind_direction']=mean
#data['wind_direction'].fillna(mean)

#-------------------------------------------------------处理风速

#处理风速的异常值，温度大于50为观测错误
#temp = data['wind_speed/kph'].dropna()
temp = np.array(data['wind_speed/kph'])
temp=temp[~np.isnan(temp)]
temp[temp > 50] = 0
temp[temp < 0] = 0
mean = np.mean(temp)
data.loc[data['wind_speed/kph'] > 50, 'wind_speed/kph'] = mean
data.loc[data['wind_speed/kph'] < 0, 'wind_speed/kph'] = mean
data.loc[data['wind_speed/kph'].isnull(),'wind_speed/kph']=mean

#data['wind_speed/kph'].fillna(mean)
print(data.info())
train_data=data
#--------------------------------gao添加----------------------------------------
#------------------------------处理NO2
print("-----------------------NO2")
#print(data.corr())
#print (train_data.info())
#print(train_data.head())
data=pd.DataFrame(data,columns=['NO2','temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'])
NO2_notnull=pd.DataFrame(data.loc[data.NO2.notnull()])
#使用随机森林时候，不能存在缺失值，否则会出现以下错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
#所以在选用NO2_notnull（）作为随机森林的训练数据时，drop掉出现空的数据，因为NO2_isnull（可以理解为预测数据）也存在其他属性为空的值，所以采用线性插值来填补数据
#print (NO2_notnull.info())
NO2_notnull.dropna(inplace=True)
#print (NO2_notnull.info())
NO2_isnull=data.loc[data.NO2.isnull()]
NO2_isnull_without_NO2=pd.DataFrame(NO2_isnull,columns=['temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'])
#NO2_isnull['humidity','temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'].dropna(inplace=True)
#NO2_isnull_without_NO2.dropna(inplace=True)s
#为了使NO2——isnull的其他数据不出现为空的情况，要么将其补全，要么drop掉，但是如果使用NO2_isnull_without_NO2.dropna(inplace=True)，则
# 在train_data.NO2.isnull（）的补全时，无法对应到相同的维度
NO2_isnull_without_NO2= NO2_isnull_without_NO2.interpolate()
print(NO2_isnull_without_NO2.info())
NO2_isnull_without_NO2.loc[NO2_isnull_without_NO2.CO.isnull()]=NO2_isnull_without_NO2.CO.mean()
NO2_isnull_without_NO2.loc[NO2_isnull_without_NO2.SO2.isnull()]=NO2_isnull_without_NO2.SO2.mean()
NO2_isnull_without_NO2.loc[NO2_isnull_without_NO2['PM2.5'].isnull()]=NO2_isnull_without_NO2['PM2.5'].mean()
NO2_isnull_without_NO2.loc[NO2_isnull_without_NO2.PM10.isnull()]=NO2_isnull_without_NO2.PM10.mean()
NO2_isnull_without_NO2.loc[NO2_isnull_without_NO2.O3.isnull()]=NO2_isnull_without_NO2.O3.mean()
#print(data.head())
X=NO2_notnull.values[:,1:]
Y=NO2_notnull.values[:,0]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
#predict_NO2=rfr.predict(NO2_isnull.values[:,1:])
predict_NO2=rfr.predict(NO2_isnull_without_NO2)
train_data.loc[train_data.NO2.isnull(),'NO2']=predict_NO2
print (pd.DataFrame(train_data).info())

#---------------------------------处理CO的缺失值------------------------------------------------
train_data=pd.DataFrame(train_data)
print ("CO\n")
print (train_data.info())
#print(train_data.head())
data=pd.DataFrame(train_data,columns=['CO','temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','SO2','PM2.5','PM10','O3'])
CO_notnull=pd.DataFrame(data.loc[data.CO.notnull()])
#使用随机森林时候，不能存在缺失值，否则会出现以下错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
#print (CO_notnull.info())
CO_notnull.dropna(inplace=True)
#print("xunlianji")
#print (CO_notnull.info())
CO_isnull=data.loc[data.CO.isnull()]
CO_isnull_without_CO=pd.DataFrame(CO_isnull,columns=['temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','SO2','PM2.5','PM10','O3'])
#print("ceshiji")
#print(CO_isnull_without_CO.info())
#NO2_isnull['humidity','temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'].dropna(inplace=True)
#NO2_isnull_without_NO2.dropna(inplace=True)
#为了使NO2——isnull的其他数据不出现为空的情况，要么将其补全，要么drop掉，但是如果使用NO2_isnull_without_NO2.dropna(inplace=True)，则
# 在train_data.NO2.isnull（）的补全时，无法对应到相同的维度
CO_isnull_without_CO= CO_isnull_without_CO.interpolate()
#在处理co时，线性插值仍然出现上述问题，经过打印发现是由于PM10存在三个缺失直，因此通过使用均值填充补齐
CO_isnull_without_CO.loc[CO_isnull_without_CO.PM10.isnull()]=CO_isnull_without_CO.PM10.mean()
#print("rrr")
print(CO_isnull_without_CO.info())
#print(data.head())
X=CO_notnull.values[:,1:]
Y=CO_notnull.values[:,0]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
#predict_NO2=rfr.predict(NO2_isnull.values[:,1:])

predict_CO=rfr.predict(CO_isnull_without_CO)
train_data.loc[train_data.CO.isnull(),'CO']=predict_CO
print (pd.DataFrame(train_data).info())
#----------------------------------------------处理SO2--------------------------------------------------
print("--------------------------------------处理SO2--------------------------------------------")

train_data=pd.DataFrame(train_data)
print ("-----------------SO2\n")
print (train_data.info())
#print(train_data.head())
data=pd.DataFrame(train_data,columns=['SO2','temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','PM2.5','PM10','O3'])
SO2_notnull=pd.DataFrame(data.loc[data.SO2.notnull()])
#使用随机森林时候，不能存在缺失值，否则会出现以下错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
#print (SO2_notnull.info())
SO2_notnull.dropna(inplace=True)
#print("xunlianji")
#print (SO2_notnull.info())
SO2_isnull=data.loc[data.SO2.isnull()]
SO2_isnull_without_SO2=pd.DataFrame(SO2_isnull,columns=['temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','PM2.5','PM10','O3'])
#print("ceshiji")
#print(SO2_isnull_without_SO2.info())
#NO2_isnull['humidity','temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'].dropna(inplace=True)
#NO2_isnull_without_NO2.dropna(inplace=True)
#为了使NO2——isnull的其他数据不出现为空的情况，要么将其补全，要么drop掉，但是如果使用NO2_isnull_without_NO2.dropna(inplace=True)，则
# 在train_data.NO2.isnull（）的补全时，无法对应到相同的维度
SO2_isnull_without_SO2= SO2_isnull_without_SO2.interpolate()
#在处理co时，线性插值仍然出现上述问题，经过打印发现是由于PM10存在三个缺失直，因此通过使用均值填充补齐
#SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2.PM10.isnull()]=SO2_isnull_without_SO2.PM10.mean()
#print("线性插值后是否仍然存在缺失")
#print(SO2_isnull_without_SO2.info())
SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2.PM10.isnull()]=SO2_isnull_without_SO2.PM10.mean()
SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2['PM2.5'].isnull()]=SO2_isnull_without_SO2['PM2.5'].mean()
SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2.O3.isnull()]=SO2_isnull_without_SO2.O3.mean()
#print(data.head())
X=SO2_notnull.values[:,1:]
Y=SO2_notnull.values[:,0]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
#predict_NO2=rfr.predict(NO2_isnull.values[:,1:])
predict_SO2=rfr.predict(SO2_isnull_without_SO2)
train_data.loc[train_data.SO2.isnull(),'SO2']=predict_SO2
print (pd.DataFrame(train_data).info())





#---------------------------------------------使用随机森林处理PM2.5缺失直
print("--------------------------------------处理PM2.5--------------------------------------------")

train_data=pd.DataFrame(train_data)
print ("-----------------PM2.5\n")
print (train_data.info())
#print(train_data.head())
data=pd.DataFrame(train_data,columns=['PM2.5','temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','SO2','PM10','O3'])
PM25_notnull=pd.DataFrame(data.loc[data['PM2.5'].notnull()])
#使用随机森林时候，不能存在缺失值，否则会出现以下错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
#print (PM25_notnull.info())
PM25_notnull.dropna(inplace=True)
#print("xunlianji")
#print (PM25_notnull.info())
PM25_isnull=data.loc[data['PM2.5'].isnull()]
PM25_isnull_without_PM25=pd.DataFrame(PM25_isnull,columns=['temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','SO2','PM10','O3'])
#print("ceshiji")
#print(PM25_isnull_without_PM25.info())
#NO2_isnull['humidity','temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'].dropna(inplace=True)
#NO2_isnull_without_NO2.dropna(inplace=True)
#为了使NO2——isnull的其他数据不出现为空的情况，要么将其补全，要么drop掉，但是如果使用NO2_isnull_without_NO2.dropna(inplace=True)，则
# 在train_data.NO2.isnull（）的补全时，无法对应到相同的维度
PM25_isnull_without_PM25= PM25_isnull_without_PM25.interpolate()
#在处理co时，线性插值仍然出现上述问题，经过打印发现是由于PM10存在三个缺失直，因此通过使用均值填充补齐
#SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2.PM10.isnull()]=SO2_isnull_without_SO2.PM10.mean()
#print("线性插值后是否仍然存在缺失")
#print(PM25_isnull_without_PM25.info())
PM25_isnull_without_PM25.loc[PM25_isnull_without_PM25.PM10.isnull()]=PM25_isnull_without_PM25.PM10.mean()
# SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2['PM2.5'].isnull()]=SO2_isnull_without_SO2['PM2.5'].mean()
PM25_isnull_without_PM25.loc[PM25_isnull_without_PM25.O3.isnull()]=PM25_isnull_without_PM25.O3.mean()
#print(data.head())
X=PM25_notnull.values[:,1:]
Y=PM25_notnull.values[:,0]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
#predict_NO2=rfr.predict(NO2_isnull.values[:,1:])
predict_PM25=rfr.predict(PM25_isnull_without_PM25)
train_data.loc[train_data['PM2.5'].isnull(),'PM2.5']=predict_PM25
print (pd.DataFrame(train_data).info())

#-----------------------------处理PM10
print("--------------------------------------处理PM10--------------------------------------------")

train_data=pd.DataFrame(train_data)
print ("-----------------PM10\n")
print (train_data.info())
#print(train_data.head())
data=pd.DataFrame(train_data,columns=['PM10','temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','PM2.5','SO2','O3'])
PM10_notnull=pd.DataFrame(data.loc[data.PM10.notnull()])
#使用随机森林时候，不能存在缺失值，否则会出现以下错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
print (PM10_notnull.info())
PM10_notnull.dropna(inplace=True)
print("xunlianji")
print (PM10_notnull.info())
PM10_isnull=data.loc[data.PM10.isnull()]
PM10_isnull_without_PM10=pd.DataFrame(PM10_isnull,columns=['temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','PM2.5','SO2','O3'])
print("ceshiji")
print(PM10_isnull_without_PM10.info())
#NO2_isnull['humidity','temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'].dropna(inplace=True)
#NO2_isnull_without_NO2.dropna(inplace=True)
#为了使NO2——isnull的其他数据不出现为空的情况，要么将其补全，要么drop掉，但是如果使用NO2_isnull_without_NO2.dropna(inplace=True)，则
# 在train_data.NO2.isnull（）的补全时，无法对应到相同的维度
PM10_isnull_without_PM10= PM10_isnull_without_PM10.interpolate()
#在处理co时，线性插值仍然出现上述问题，经过打印发现是由于PM10存在三个缺失直，因此通过使用均值填充补齐
#SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2.PM10.isnull()]=SO2_isnull_without_SO2.PM10.mean()
print("线性插值后是否仍然存在缺失")
print(PM10_isnull_without_PM10.info())
PM10_isnull_without_PM10.loc[PM10_isnull_without_PM10.O3.isnull()]=PM10_isnull_without_PM10.O3.mean()
#print(data.head())
X=PM10_notnull.values[:,1:]
Y=PM10_notnull.values[:,0]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
#predict_NO2=rfr.predict(NO2_isnull.values[:,1:])
predict_PM10=rfr.predict(PM10_isnull_without_PM10)
train_data.loc[train_data.PM10.isnull(),'PM10']=predict_PM10
print (pd.DataFrame(train_data).info())
#----------------------------------------------处理O3--------------------------------------------------
print("--------------------------------------处理O3--------------------------------------------")
train_data=pd.DataFrame(train_data)
print ("-----------------O3\n")
print (train_data.info())
#print(train_data.head())
data=pd.DataFrame(train_data,columns=['O3','temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','PM2.5','SO2','PM10'])
O3_notnull=pd.DataFrame(data.loc[data.O3.notnull()])
#使用随机森林时候，不能存在缺失值，否则会出现以下错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
#print (PM10_notnull.info())
O3_notnull.dropna(inplace=True)
#print("xunlianji")
#print (PM10_notnull.info())
O3_isnull=data.loc[data.O3.isnull()]
O3_isnull_without_O3=pd.DataFrame(O3_isnull,columns=['temperature','pressure','humidity','wind_direction','wind_speed/kph','NO2','CO','PM2.5','SO2','PM10'])
#print("ceshiji")
print(O3_isnull_without_O3.info())
#NO2_isnull['humidity','temperature','pressure','humidity','wind_direction','wind_speed/kph','CO','SO2','PM2.5','PM10','O3'].dropna(inplace=True)
#NO2_isnull_without_NO2.dropna(inplace=True)
#为了使NO2——isnull的其他数据不出现为空的情况，要么将其补全，要么drop掉，但是如果使用NO2_isnull_without_NO2.dropna(inplace=True)，则
# 在train_data.NO2.isnull（）的补全时，无法对应到相同的维度
O3_isnull_without_O3= O3_isnull_without_O3.interpolate()
#在处理co时，线性插值仍然出现上述问题，经过打印发现是由于PM10存在三个缺失直，因此通过使用均值填充补齐
#SO2_isnull_without_SO2.loc[SO2_isnull_without_SO2.PM10.isnull()]=SO2_isnull_without_SO2.PM10.mean()
print("线性插值后是否仍然存在缺失")
print(O3_isnull_without_O3.info())
#O3_isnull_without_O3.loc[O3_isnull_without_O3.O3.isnull()]=PM10_isnull_without_PM10.O3.mean()
#print(data.head())
X=O3_notnull.values[:,1:]
Y=O3_notnull.values[:,0]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
#predict_NO2=rfr.predict(NO2_isnull.values[:,1:])
predict_O3=rfr.predict(O3_isnull_without_O3)
train_data.loc[train_data.O3.isnull(),'O3']=predict_O3
print (pd.DataFrame(train_data).info())
#--------------------------------------------------------处理天气数据
weather_mode=(train_data['weather']).mode()
train_data.loc[train_data.weather.isnull(),'weather']=weather_mode[0]
print (pd.DataFrame(train_data).info())

#----------------------------------------    3-   插值和进行排序 （标准化放到转h5文件中）       ------------------------------------------#
data = train_data.sort_values(by = ['utc_time', 'stationId_aq'])
print("end:",data.info())
#data = data.sort_values(by = ['utc_time', 'stationId_aq'])

#对于新爬去下来的数据， 将不需要的站点数据去除掉     因为发现有限地方是线性插值不上的。      这里还有按照 一定顺序排列行 (因为训练时，站点位置是固定次序的)。

#使用reindex就可以对行顺序，内部指定clumn就可以对列上调顺序    这里的想法是构造一个新的index需要的列表，之后作为reindex项填充进去，实现排序。   不用重新排了，去掉没用的，已经对了


#data = data.iloc[:12].reindex(['CD1','BL0','GR4','MY7','HV1','GN3','GR9','LW2','GN0','KF1','CD9','ST5','TH4']) #使用reindex就可以对行顺序，内部指定clumn就可以对列上调顺序
data.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/final_merge_aq_grid_meo_deal_new.csv',index=False)


























#
#
# import pandas as pd
# import numpy as np
#
# #----------------------------------------    1-   将站点数据和气象数据进行合并------------------------------------------#
# '''
# 这里调用其他文件里面的函数来是进行数据的合并
# 1.先用 merge_meo_grid做下整体插值，获得所有站点的气象数据的方式,能得到每个站点再当前时刻该有的气象数据结构
#
# merge_aq_meo是不需要的，因为数据里面没有气象站数据， 第一次合并搞好了，之前整了一次数据乌龙，经纬度位置不对应出现问题了
# 2.再作merge_aq_gird_meo就完全搞定了，将第一步处理的结果数据保存下来。。
# '''
# #再那两个融合文件里，包括  dara_preparation进行时间数据调整
# #----------------------------------------    2-   去异常值和线性插值        ------------------------------------------#
# data=pd.read_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/final_merge_aq_grid_meo2.csv")
# print(data.info())
# #处理温度的异常值，温度大于100为观测错误
# #    先统计出均值，之后用均质进行  填充。。。   理好逻辑，初始赋0对本体是没有影响的。
# temp = np.array(data['temperature'])
# temp[temp > 60] = 0
# mean = np.mean(temp)
# data.loc[data['temperature'] > 100, 'temperature'] = mean
# #data['temperature'].fillna(mean)
# print(data.info())
# #print(data['temperature'])
#
# #处理压力的异常值，温度大于2000为观测错误
# #temp = data['pressure'].dropna()
# temp = np.array(data['pressure'])
# temp[temp > 2000] = 0
# mean = np.mean(temp)
# data.loc[data['pressure'] > 2000, 'pressure'] = mean
# #data['pressure'].fillna(mean)
# #处理湿度的异常值，温度大于100为观测错误
# #temp = data['humidity'].dropna()
# temp = np.array(data['humidity'])
# temp[temp > 100] = 0
# mean = np.mean(temp)
# data.loc[data['humidity'] > 100, 'humidity'] = mean
# #data['humidity'].fillna(mean)
# # print(data['humidity'])
#
# #处理风向的异常值,风向大于360为观测错误，但是风向越有5000异常值
# #temp = data['wind_direction'].dropna()
# temp = np.array(data['wind_direction'])
# temp[temp > 360] = 0
# mean = np.mean(temp)
# data.loc[data['wind_direction'] > 360, 'wind_direction'] = mean
# #data['wind_direction'].fillna(mean)
# print(data['wind_direction'])
#
#
# #处理风速的异常值，温度大于50为观测错误
# #temp = data['wind_speed/kph'].dropna()
# temp = np.array(data['wind_speed/kph'])
# temp[temp > 50] = 0
# mean = np.mean(temp)
# data.loc[data['wind_speed/kph'] > 50, 'wind_speed/kph'] = mean
# #data['wind_speed/kph'].fillna(mean)
# print(data.info())
#
# temp = np.array(data['PM2.5'])
# temp[temp > 50] = 0
# mean = np.mean(temp)
# data.loc[data['PM2.5'] > 100, 'PM2.5'] = mean
# #data['PM2.5'].fillna(mean)
# print(data.info())    #info是一种介绍信息， head是得到前几行数据
# #----------------------------------------    3-   插值和进行排序 （标准化放到转h5文件中）       ------------------------------------------#
#
# data = data.sort_values(by = ['utc_time', 'stationId_aq'])
#
# #对于新爬去下来的数据， 将不需要的站点数据去除掉     因为发现有限地方是线性插值不上的。      这里还有按照 一定顺序排列行 (因为训练时，站点位置是固定次序的)。
# data = data.interpolate()
# data['O3'][0]=120
# data['O3'][1]=118
# data['O3'][2]=116
# #使用reindex就可以对行顺序，内部指定clumn就可以对列上调顺序    这里的想法是构造一个新的index需要的列表，之后作为reindex项填充进去，实现排序。   不用重新排了，去掉没用的，已经对了
#
#
# #data = data.iloc[:12].reindex(['CD1','BL0','GR4','MY7','HV1','GN3','GR9','LW2','GN0','KF1','CD9','ST5','TH4']) #使用reindex就可以对行顺序，内部指定clumn就可以对列上调顺序
# data.to_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/final_merge_aq_grid_meo_deal_new.csv',index=False)