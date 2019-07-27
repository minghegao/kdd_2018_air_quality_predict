
import  pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
import numpy as np

# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import math
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
from sklearn.preprocessing import MinMaxScaler  # 这是标准化处理的语句，很方便，里面有标准化和反标准化。。

np.random.seed(1337)  # for reproducibility
# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH  # 配置的环境
T = 24  # number of time intervals in one day   一天的周期迭代次数

lr = 0.0001  # learning rate
len_closeness = 6  # length of closeness dependent sequence   考虑的相邻的迭代次数
len_period = 1  # length of peroid dependent sequence       以相邻周期四个作为预测趋势
len_trend =4   # length of trend dependent sequence    以前面4个作为趋势性
nb_residual_unit = 6  # number of residual units   残差单元数量

nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days    使用10天数据进行测试
days_test = 10
len_test = T * days_test  # 测试用的时间戳数量
map_height, map_width = 35, 12  # grid size   每个代表流量意义的格点图的大小为16*8
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81  # 共有81个基于网格点的区域， 每个区域至少有1个自行车站
# m_factor 计算得到影响因素   影响因子的具体计算为什么这样算
path_result = 'RET'
path_model = 'MODEL'

'''
下面的程序将执行预测人物，  我打算把 大的数据集加载过来，找到其中

'''


def build_model(external_dim):
    # 创建模型时   首先指定进行组合时的参数    将配置分别放进不同的区域中（就是相近性的长度 ，周期性的长度等）
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_period > 0 else None
    '''
    趋势性的数据暂时不要
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None
    '''
    # 根据不同的配置定义残差神经网络模型  这个stresnet是定义好的，传入关于不同方面的配置，最后会返回根据参数组合好的模型
    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)  # 接下来 定义学习率和损失函数值
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    #model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def main():
    # 加载预测需要用到的数据
    print("loading data...")
    X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
        len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=True)

    for _X in X_train:
        print('theshape  ', _X.shape, )

    model = build_model(external_dim)
    fname_param = '/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/scripts/AirPrediction/MODEL/c{len}.p{per}.t{trend}.resunit{res_num}.lr{leaing_ra}.cont.best.h5'.format(
        len=len_closeness, per=len_period, trend=len_trend, res_num=nb_residual_unit, leaing_ra=lr)
    model.load_weights(fname_param)
    # 开始使用模型进行预测    这里有毒，草，   这里需要的输入是三个输入
    print('Y_train[0]:', type(Y_train))

    '''
    这里的问题困惑了我很长时间，也怪我对kreas和 神经网络的训练机制不够理解的通透，
    第一点：这里 因为自己创建的模型的输入是有批次数的，所以对应进行预测的时候 输入的数据也得有批次数（kreas的预测、评估、训练都是按批次的），对于predict预测也是这样的，  所以这里我是限定了维度，包装成了需要的
    输入的形式，这是根据X的形式包装的一个特征数据对应的输出预测。
    第二点: 在于对批次的理解上，和师姐交流了下，对于 图像数据是4唯的网络输入（批次数，通道，长度，宽度），对于lstm是3维的输入（批次数，记忆体长度..），其中一次训练时候
    是按批次的，相当于加了个维度，按正常的方式由前往后跑，一个批次每个数据都会得到一个损失值，得到的损失值进行批次内sum合并之后再进行一次传播，所以理明白 模型的训练是以
    一个批次的数据为单位进行训练的，  跑模型时候批次的作用体现在扩展维度上，  批次内数据也就是一个高唯独内的数据。

    还有构造的模型中根本没有考虑批次的事情，将批次的融入是kreas中的fit所作的事情， 可以看到模型中根本就没有batch_size的事情。
    第三点：kreas中模型的输入不像TF是按名 feed传值的，kreas中的方式是  按照计算流图找到开始的传入将数据传给神经网络，对应放到输入的位置，自己传入所有数据，在fit中指定
    好批次就好，数据会按批次训练。。。    
    其实这里有个问题是外部因素的融入我只融入了一天一个唯独的节假日影响（是以后一个时刻点所对应的数据），

    '''
    # ========================      进行预测      ===========================#
    #  下面是一种模式，已经掌握了其思想
    #循环记录下，多个时间点下的得分强况
    the_predict_index = 180
    the_time_list=[]
    smape_score_list=[]
    #获取长期的得分变化
    for the_predict_index in range(0,the_predict_index,5):
        y_pred = model.predict([X_train[0][the_predict_index].reshape(1, len_closeness, 35, 12),
                                X_train[1][the_predict_index].reshape(1, len_period, 35, 12),
                                X_train[2][the_predict_index].reshape(1, len_trend, 35, 12),
                                X_train[3][the_predict_index].reshape(1, external_dim)], batch_size=1)
        #print('成功得到时刻点', timestamp_train[the_predict_index], '的一个预测结果：', y_pred.shape)
        the_last_list = []
        # 现在将不同时刻点的数据按列表方式进行组合成一个二维的，准备存到csv文件中去。
        # ========================      遍历和组装      ===========================#
        # 下面的方式 是一种遍历， array中的特征数据 全放在一个列表中去，   发现直接时间拼接的方式不对,最后也终于攻克了，就是列表的拼接方式。。。
        for i in range(len(y_pred[0])):
            for ii in y_pred[0][i]:
                # 这里必须写三步，这是我的新发现。 不这样写 list就变成None了，不可思议爱
                ii_list = ii.tolist()
                ii_list.append(timestamp_train[the_predict_index + i])
                the_last_list.append(ii_list)
                # print(ii_list.append(timestamp_train[i][0]))
                # the_last_list.append()

        # print(y_pred[0][0][0].tolist().append('dsa'))
        # print('the_last_list  length:',the_last_list)
        pred_array = pd.DataFrame(the_last_list, columns=['PM2.5', 'PM10', 'O3', 'utc_time'])
        pred_array.to_csv(
            '/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/the_predict_data.csv',
            index=False)

        the_time,smape_score=offline_score(pred_array,timestamp_train[the_predict_index])
        #print('the time:',the_time,'   the score:',smape_score)
        the_time_list.append(the_time)
        smape_score_list.append(smape_score)

    print(' 平均下来的评分为：',sum(smape_score_list)/len(smape_score_list))
    show_score=pd.DataFrame(smape_score_list,index=dict(map(lambda x,y:[x,y], the_time_list,smape_score_list)).keys())
    print(show_score)







def smape(actual, predicted):
    #print('actual',actual)
    #print('predicted',predicted)
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    return 2*np.mean(np.divide(a, b, out=np.    zeros_like(a),where=b!= 0,casting='unsafe'))

def offline_score(ok_filename,the_time):




    #---------------------------------------------------    比较简便的反标准化的过程    ----------------------------------------------------------#
    sta_origion_data=pd.read_csv('/home/fly/PycharmProjects/DeepST-KDD for_train/for_submit_data/gao_data/final_merge_aq_grid_meo_with_weather_sort.csv')

    read_csvfile=pd.read_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/final_merge_aq_grid_meo_deal_weather.csv')
    # ========================      对数据读取并标准化处理下（标准化和反标准化需要一个去适配）      ===========================#
    scaler = MinMaxScaler(feature_range=(-1, 1))  # 数值限定到-1到1之间 看出来了，这是在同一个scaler下，将tranform转换后的数据，使用reverse还原回来，scaler有记忆的，所以在这里没用
    #fit之前先挑选出全都是数值型的列     这里针对数值型进行按序列标准化之后再进行拼接，这里在使用fit_transform之后列的标题会丢掉，我又重加了进去。。。   标准化时候用了天气，但是训练时候没用天气，但是这里天气要加

    '''
    有时候不明白已经要加上fit_transform过程，  反标准化化的结果才正确，  也是无奈了，算了，如果不进行正标准化的过程，得到的结果好像会翻倍偶
    '''
    scaler.fit(sta_origion_data[['PM2.5', 'PM10', 'O3']])  # Compute the minimum and maximum to be used for later scaling 这是个fit过程， 这个过程会 计算以后用于缩放的平均值和标准差， 记忆下来
    data1 = pd.DataFrame(scaler.fit_transform(read_csvfile[['PM2.5', 'PM10', 'O3']]),columns=['PM2.5', 'PM10', 'O3'])  # Fit to data, then transform it. 使用记忆的数据进行转换
    data2 = read_csvfile[['stationId_aq', 'utc_time']]
    #frames = [data1, data2]  #在列上进行拼接。。。    还有很多链接方式的选择，左右、外链接等方式，全在书上，方式很多。。。
    # read_csvfile = pd.concat(frames, axis=1)
    # pred_array=pd.DataFrame(read_csvfile)
    # # ========================      插曲-借助进行反标准化 （这里很大的问题是没有天气信息，我随便给的天气，发现预测结果还不对，对着那）     ===========================#
    ok_filename = pd.read_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/the_predict_data.csv')
    # scaler.fit(sta_origion_data[['PM2.5', 'PM10',
    #                              'O3']])  # Compute the minimum and maximum to be used for later scaling 这是个fit过程， 这个过程会 计算以后用于缩放的平均值和标准差， 记忆下来
    # #
    pred_array = pd.DataFrame(scaler.inverse_transform(ok_filename[['PM2.5', 'PM10', 'O3']]),columns=['PM2.5', 'PM10', 'O3'])  #使用记忆的数据反转换
    pred_array1 = ok_filename[['utc_time']]
    frames1 = [pred_array, pred_array1]
    read_csvfile1 = pd.concat(frames1, axis=1)
    #pred_array=pd.DataFrame(read_csvfile1,columns=['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph','weather', 'NO2', 'CO', 'SO2', 'PM2.5', 'PM10', 'O3','utc_time'])
    pred_array=pd.DataFrame(read_csvfile1)
    pred_array.to_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/the_predict_data_renormation.csv',index=False)
    #print(pred_array)


    #-----------   拼接得到站点----------------------#
    aq_data = pd.read_csv("/home/fly/PycharmProjects/DeepST-KDD/data/4-14_new_data/2017_data/final_merge_aq_grid_meo_with_weather4-14.csv")
    predict_data = pd.read_csv("/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/the_predict_data_renormation.csv")
    #提取  需要的信息表中  需要的 列名顺序
    df_aq = pd.DataFrame(aq_data,index=None,columns=['stationId_aq'])
    df_aq_2100 = df_aq.loc[0:1749,:]
    #对所有列名进行改名字，将 test_id上面每一列改成提交需要的数据样式
    df_predict = pd.DataFrame(predict_data,index=None,columns=['PM2.5','PM10','O3','utc_time'])
    #print('df_predict',df_predict)
    #use_df_predict=df_predict.loc[0:623,:]
    df_merge = pd.concat([df_aq_2100,df_predict.loc[:,:]],axis=1,join='outer')#  完全的外部列拼接方式
    #print('df_merge:',df_merge)







    #------------------------------------------------      获取对应时刻真实的数据        ----------------------------------------------#
    history_file = pd.read_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/final_merge_aq_grid_meo2.csv')
    #print(pred_array['utc_time'][0:12])
    #print(history_file[history_file['utc_time'].isin([str(datetime.datetime.strptime(str(i[2:12]), '%Y%m%d%H'))  for i in pred_array['utc_time']])])
    true_array=history_file[history_file['utc_time'].isin([str(datetime.datetime.strptime(str(i[2:12]), '%Y%m%d%H'))  for i in pred_array['utc_time']])]
    #print('true_array:',true_array)



    #------------------------------------------------      准备排序和对应位置计算        ----------------------------------------------#
    pred_array =df_merge.sort_values(by = ['utc_time', 'stationId_aq'])[['PM2.5','PM10','O3']]
    true_array=true_array.sort_values(by = ['utc_time', 'stationId_aq'])[['PM2.5','PM10','O3']]

    #print('aa   true_array:',true_array)
    #print('aa   pred_array:',pred_array)

    '''
    进行比对时候排除那些为空 的   实际结果，这个官网上有说明，如果为空，就把那一行都去掉。  
    '''
    pred_array_list=np.array(pred_array).tolist()
    true_array_list=np.array(true_array).tolist()
    the_new_true_array_lis=[]
    the_new_pred_array_list=[]
    #print(true_array_list)

    '''
    这里有个坑，    nan类型不能使用普通的is in等方法匹配，而必须使用pd.isnull这类方法来匹配才匹配得到
    '''
    #print('::::::::::',true_array_list[5][1])
    if pd.isnull(true_array_list[5][1]):
        print('你吗')
    for i in range(len(true_array_list)):
         flag=1
         for j in true_array_list[i]:
             if(pd.isnull(j)==True):
                 flag=0
         if flag==1:
             the_new_true_array_lis.append(true_array_list[i])
             the_new_pred_array_list.append(pred_array_list[i])




    #----------------------------散点图绘制----------------------#
    import  matplotlib.pyplot as plt
    fig=plt.figure(num='Beijing   time：'+str(the_time),figsize=(100,100))
    ax_pm25=fig.add_subplot(2,2,1)
    ax_pm25.set_title('pm 2.5')
    ax_pm10=fig.add_subplot(2, 2, 2)
    ax_pm10.set_title('pm 10')
    ax_O3=fig.add_subplot(2, 2, 3)
    ax_O3.set_title('O3')
    type1=ax_pm25.scatter(np.arange(len(the_new_pred_array_list)),[i[0]*2  for i in the_new_pred_array_list],color='r')
    type2=ax_pm25.scatter(np.arange(len(the_new_true_array_lis)),[i[0]  for i in the_new_true_array_lis],color='g',alpha=0.5)#真实的透明点
    ax_pm25.legend((type1, type2), (u'pred', u'true'), loc=2)


    type3=ax_pm10.scatter(np.arange(len(the_new_pred_array_list)),[i[1]  for i in the_new_pred_array_list],color='r')
    type4=ax_pm10.scatter(np.arange(len(the_new_true_array_lis)),[i[1]  for i in the_new_true_array_lis],color='g',alpha=0.5)#真实的透明点
    ax_pm10.legend((type3, type4), (u'pred', u'true'), loc=2)

    type5 = ax_O3.scatter(np.arange(len(the_new_pred_array_list)), [i[2] for i in the_new_pred_array_list], color='r')
    type6 = ax_O3.scatter(np.arange(len(the_new_true_array_lis)), [i[2] for i in the_new_true_array_lis], color='g', alpha=0.5)  # 真实的透明点
    ax_pm10.legend((type5, type6), (u'pred', u'true'), loc=2)

    plt.show()
    # ------------------------------------------------------#

    #删除  true_array中为空的行

    #北京线下得分有点低阿         北京的分值贼高，  伦敦的还好，  最好将数据精准化，  用的少的化，太误导网络了。


    print('时间点：',the_time,'   smape  得分是：'+str(smape(the_new_true_array_lis,the_new_pred_array_list)))
    return the_time,smape(the_new_true_array_lis,the_new_pred_array_list)


if __name__ == '__main__':
    main()


