'''
根据模板合并数据
beijing_17_18_aq.csv
beijing_17_18_meo.csv
模板
merage_aq_meo_beijing_template.csv
输出文件
assets/output/merage_aq_meo/merage_aq_meo_beijing.csv
'''

import pandas as pd
from math import*
import matplotlib.pyplot as plt
import numpy as np

#对文件 按照序号聚集 再一起，并提交#。。。

# 读入数据文件
predict_data = pd.read_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_london_for_predict/for_submit_data/submit_date.csv")
df_predict_data = pd.DataFrame(predict_data,index=None,columns=['test_id','PM2.5','PM10'])
#先修改命名 进行排序，排完序之后再将命名改回来
for index,row in df_predict_data.iterrows():
    if index<624:
        value = row['test_id']# = row['test_id']+"#"+str(index)
        print('vale:   ',value)
        num_value = value.split('#')[1]
        prefix_value = value.split('#')[0]
        if len(num_value) == 1:
            result_value = prefix_value + '#0'+str(num_value)
            df_predict_data['test_id'][index] = result_value
df_predict_data = df_predict_data.sort_values(by='test_id')
for index,row in df_predict_data.iterrows():
    if index<624:
        #print(index)
        value = row['test_id']
        num_value_str = value.split('#')[1]
        prefix_value = value.split('#')[0]
        if num_value_str[0] == '0':
            df_predict_data['test_id'][index] = prefix_value + '#'+num_value_str[1]

#print(df_predict_data)
df_predict_data.to_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_london_for_predict/for_submit_data/submit_date_final.csv")


#--------------------------------      按最终要求次序的提交方式         -----------------------------------—#

predict_data = pd.read_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_london_for_predict/for_submit_data/submit_date_final.csv")
pd_final=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final1=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final2=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final3=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final4=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final5=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final6=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final7=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final8=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final9=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final10=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final11=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final12=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final13=pd.DataFrame(columns=['Unnamed: 0','test_id','PM2.5','PM10'])

for index,row in predict_data.iterrows():
    if index<624:
        if row['test_id'][:4]=='CD1#':
             pd_final1=pd_final1.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='BL0#':
             pd_final2=pd_final2.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='GR4#':
               pd_final3=pd_final3.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='MY7#':
            pd_final4=pd_final4.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='HV1#':
            pd_final5=pd_final5.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='GN3#':
            pd_final6=pd_final6.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='GR9#':
            pd_final7=pd_final7.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='LW2#':
            pd_final8=pd_final8.append(row)  #把一扩展进去
        if row['test_id'][:4]=='GN0#':
            pd_final9=pd_final9.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='KF1#':
            pd_final10=pd_final10.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='CD9#':
            pd_final11=pd_final11.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='ST5#':
            pd_final12=pd_final12.append(row)  #把一行扩展进去
        if row['test_id'][:4]=='TH4#':
            pd_final13=pd_final13.append(row)  #把一行扩展进去
# b=pd.Series([6,7,8,9],index=['Unnamed: 0','test_id','PM2.5','PM10'])
pd_final=pd.concat([pd_final1,pd_final2,pd_final3,pd_final4,pd_final5,pd_final6,pd_final7,pd_final8,pd_final9,pd_final10,pd_final11,pd_final12,pd_final13,],axis=0)

pd_final.to_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_london_for_predict/for_submit_data/submit_test.csv")