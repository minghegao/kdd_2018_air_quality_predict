#----------------  想办法  -----------------#
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

#--------------------------------    1     ----------------------------------#
# 读入数据文件
aq_data = pd.read_csv("/home/fly/PycharmProjects/DeepST-KDD_london/london_data/origion_data/final_merge_aq_grid_meo3.csv")
predict_data = pd.read_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_london_for_predict/for_submit_data/the_predict_data_restation.csv")
#提取  需要的信息表中  需要的 列名顺序
df_aq = pd.DataFrame(aq_data,index=None,columns=['stationId_aq'])
df_aq.rename(columns={'stationId_aq':'test_id'},inplace=True)
df_aq_2100 = df_aq.loc[0:623,:]
#对所有列名进行改名字，将 test_id上面每一列改成提交需要的数据样式
data_index=0
for index,row in df_aq_2100.iterrows():
    row['test_id'] = row['test_id']+"#"+str(data_index)
    if((index+1)%13==0):
        data_index+=1
#print(len(df_aq_2101))


'''
思想
  拼接是人工操作表
'''
#--------------------------------    2     ----------------------------------#
#  提取  预测表中需要的信息，  再列上进行融合    156:
df_predict = pd.DataFrame(predict_data,index=None,columns=['PM2.5','PM10'])
df_predict=df_predict.drop([i for i in range(26)])
print('df_predict',df_predict)
#use_df_predict=df_predict.loc[0:623,:]
df_merge = pd.concat([df_aq_2100,df_predict.loc[:,:]],axis=1,join='outer')

#--------------------------------    3     ----------------------------------#
df_merge
df_merge.to_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_london_for_predict/for_submit_data/submit_date.csv',index=False)

