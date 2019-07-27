from catboost  import CatBoostRegressor
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
pd.set_option('display.max_columns',5000)
'''
这里有个很重要的问题，文件名绝不能是xgboost,不然的话会把 模块文件给顶替了。改名
'''
def smape_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(np.fabs(preds - labels) / (preds + labels) * 2), False

def f1(x):
    return np.log(x+1)
def rf1(x):
    return np.exp(x)-1

def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)


def train_model(train_air='pm25'):
    '''
    使用这个函数可以训练三个模型（分别按指标）

    删除时候发现每一行都有空值 导致删空时候不让删去
    :param train_air:  指标名
    :return:
    '''
    f = h5py.File('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/'+train_air+'.h5', 'r')
    #f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value

    pm25_data = pd.DataFrame(data,columns=[str(i) for i in range(0,3386)])
    #print(pm25_data[pm25_data.ix[1:20000,:].isnull()])
    pm25_data.dropna(axis=0)

    pm25_data=pm25_data.ix[1:500000,:]

    train_X, test_X, train_Y, test_Y = train_test_split(pm25_data[[str(i) for i in range(0,3385)]], pm25_data['3385'], test_size=0.2, random_state=11)

    gbm0 = CatBoostRegressor()
    gbm0.fit(train_X, train_Y)
    test_Y1 = gbm0.predict(test_X)
    score = get_score(test_Y1, test_Y)

    #model_param['tree'] = cv_results.best_iteration
    #print(cv_results.best_iteration)

    model_file='/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/cat_'+train_air+'.model'
    with open(model_file, 'wb') as fout:
        pickle.dump(gbm0, fout)
    #print(model_param)


def model_predict(train_air='pm25'):
    f = h5py.File('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/'+train_air+'.h5', 'r')
    data = f['data'].value
    pm25_data = pd.DataFrame(data,columns=[str(i) for i in range(0,3386)])

    model_path='/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/cat_'+train_air+'.model'
    model = pickle.load(open(model_path, 'rb'))

    #打印所有数据集的时刻上的预测的分
    the_sum_score=0
    the_sum_num=0
    for i in range(600,1200):
        start_num=840*i
        end_num=(i+1)*840

        print('start_num:',start_num)
        print('end_num:',end_num)

        pm_data = pm25_data.ix[int(start_num):int(end_num), :]

        pred_PM25 = model.predict(pm_data[[str(i) for i in range(0,3385)]])

        score = get_score(pred_PM25, pm_data['3385'])
        print(str(i)+'次，计算留出集合上损失得分：', score)
        the_sum_score=the_sum_score+score
        the_sum_num=the_sum_num+1

    print('GBDT 平均得分：',the_sum_score/the_sum_num)

model_predict()