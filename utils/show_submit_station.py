import pandas as pd
import numpy as np
'''
本函数用于展示要提交的表中的所有站点名字，    以及自己预测结果中的站点名字，   用于发现两者之间的差距，从而修改名称，使得能够对应放置过去，
对了 不修改名字也行，只要顺序对就行，到时候能直接复制过去

'''
#-----------------------------------      展示自己预测结果中的站点名字    ---------------------------------------#
my_submit_station=pd.read_csv('/home/fly/PycharmProjects/DeepST-KDD/for_submit_data/submit_date.csv')
station_list=[]
for index,row in my_submit_station.iterrows():
    #print(index)
    station_list.append(row['test_id'].split('#')[0])
#去除列表中重复的数据   (使用字典的方式去重)
station_list={}.fromkeys(station_list).keys()
print(station_list)

#-----------------------------------      要提交的样例中站点名字    ---------------------------------------#
my_submit_station=pd.read_csv('/home/fly/PycharmProjects/DeepST-KDD/for_submit_data/my_submissioin.csv')
station_list=[]
for index,row in my_submit_station.iterrows():
    #print(index)
    station_list.append(row['test_id'].split('#')[0])
#去除列表中重复的数据   (使用字典的方式去重)
station_list={}.fromkeys(station_list).keys()
print(station_list)
