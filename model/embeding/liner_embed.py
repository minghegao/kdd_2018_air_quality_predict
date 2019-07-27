import pandas as pd
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle

'''
选取表现效果最好的三个模型进行融合，决定使用 lightgbm gbdt  extra_tree 三个先进行线性融合 
'''
def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)

def liner_embed(train_air='pm25'):
    f = h5py.File('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/'+train_air+'.h5', 'r')
    data = f['data'].value
    pm25_data = pd.DataFrame(data,columns=[str(i) for i in range(0,3386)])


    model_path = '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/lgb_' + train_air + '.model'
    model_lgb = pickle.load(open(model_path, 'rb'))

    model_path = '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/gbdt_' + train_air + '.model'
    model_gbdt = pickle.load(open(model_path, 'rb'))

    model_path='/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/extra_'+train_air+'.model'
    model_extra = pickle.load(open(model_path, 'rb'))

    pred_lgb_PM25_n=0.1
    pred_gbdtb_PM25_n = 0.1
    pred_extra_PM25_n = 0.1
    min_score=1
    min_score_gb=0.5
    min_score_ex=0.1
    min_score_lgb=0.4
    print('~~~~~~~~~~~~~~~~~')
    #用小数遍历效果不好           0.5633611749951315     5      1      4
    for pred_lgb_PM25_n in range(1,10):
        for pred_gbdtb_PM25_n in range(1,10):
            for pred_extra_PM25_n in range(1,10):
                print('~~~~~~~~~~~~~~~~~',pred_lgb_PM25_n,'   ',pred_gbdtb_PM25_n,'    ',pred_extra_PM25_n)
                if pred_extra_PM25_n+pred_gbdtb_PM25_n+pred_lgb_PM25_n==10:
                    the_sum_score=0
                    the_sum_num=0
                    for i in range(500,1200):
                        start_num=840*i
                        end_num=(i+1)*840

                        #print('start_num:',start_num)
                        #print('end_num:',end_num)

                        pm_data = pm25_data.ix[int(start_num):int(end_num), :]

                        pred_lgb_PM25 = model_lgb.predict(pm_data[[str(i) for i in range(0,3385)]])
                        pred_gbdt_PM25 = model_gbdt.predict(pm_data[[str(i) for i in range(0,3385)]])
                        pred_extr_PM25 = model_extra.predict(pm_data[[str(i) for i in range(0,3385)]])

                        score = get_score((pred_lgb_PM25*pred_lgb_PM25_n/10+pred_gbdt_PM25*pred_gbdtb_PM25_n/10+pred_extr_PM25*pred_extra_PM25_n/10), pm_data['3385'])
                        #print(str(i)+'次，计算留出集合上损失得分：', score)
                        the_sum_score=the_sum_score+score
                        the_sum_num=the_sum_num+1
                    #f['data'].value存放的是时间戳 上空间的流量数据
                    print('GBDT 平均得分：',the_sum_score/the_sum_num)
                    if(the_sum_score/the_sum_num<min_score):
                        min_score_gb=pred_gbdtb_PM25_n
                        min_score_ex=pred_extra_PM25_n
                        min_score_lgb=pred_lgb_PM25_n
                        min_score=the_sum_score/the_sum_num
                        print('替换！！！！')

                pred_extra_PM25_n=float(pred_extra_PM25_n+1)
            pred_gbdtb_PM25_n=float(pred_gbdtb_PM25_n+1)
        pred_lgb_PM25_n=float(pred_lgb_PM25_n+1)
    print('   ',min_score,'   ',min_score_gb,'    ',min_score_ex,'    ',min_score_lgb,'    ','')
liner_embed()