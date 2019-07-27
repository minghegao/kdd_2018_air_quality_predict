import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle
import sklearn
from keras.callbacks import Callback
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler  # 这是标准化处理的语句，很方便，里面有标准化和反标准化。。
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor

def smape_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(np.fabs(preds - labels) / (preds + labels) * 2), False

def min_max_normalize(data):
    # 归一化       数据的归一化计算，这样计算之后结果能更加适合非树模型，  但是进行归一化之后怎么反归一化得看下
    #数据量大，标准化慢
    df=data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # 做简单的平滑,试试效果如何
    return df

def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)

def NN_Model(train_air='pm25'):
    '''
        训练nn模型，并提取，倒数第二层的特征，特征的提取方法可参照：https://blog.csdn.net/hahajinbu/article/details/77982721
        进行最后一层的抽取方法是， 先训练一个nn模型model，但是要提前给每层都赋好层命名，   之后再简历一个Model,输入是上一个模型
        训练所使用到的数据，输出是上一个model的指定层名，最为输出，然后使用Model去做预测，得到输出那一层结果
        '''
    f = h5py.File(
        '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/' + train_air + '.h5',
        'r')


    # f['data'].value存放的是时间戳 上空间的流量数据
    data = f['data'].value
    print('!')
    pm25_data_all = pd.DataFrame(data, columns=[str(i) for i in range(0, 3386)])
    print('00')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_all.ix[1:400000, :])
    pm25_data_all=pd.DataFrame(scaler.fit_transform(pm25_data_all.ix[1:400000, :]),columns=[str(i) for i in range(0, 3386)])

    #print(pm25_data_all.head())
    #print('000000000000000000',pm25_data_all.columns.values.tolist())

    #pm25_data_nor = min_max_normalize(pm25_data_all)
    #print(pm25_data_nor.columns.values.tolist())




    #pm25_data_all = pm25_data_all.ix[1:300000, :]
    train_data=pm25_data_all[[str(i) for i in range(0, 3385)]]
    train_flag=pm25_data_all['3385']
    print('!!!')
    train_x, valid_x, train_y, valid_y = train_test_split(train_data,train_flag,
                                                        test_size=0.2, random_state=11)
    print('!!!!!')
    #print(train_x)

    model = Sequential()
    model.add(Dense(activation='relu', units=2000, input_dim=train_x.shape[1]))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=1000, name='Dense_2'))
    model.add(Dense(activation='tanh', units=1))
    optimizer = Adam(lr=0.00001)
    model.compile(loss='mse', optimizer=optimizer)

    # mc = ModelCheckpoint(filepath="./model/weights-improvement-{epoch:02d}-{val_auc:.2f}.h5", monitor='val_auc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=0)
    es = EarlyStopping(monitor='val_rmse', patience=10, verbose=1, mode='min')
    print('!!!aaaaaaaaaa')
    model.fit(x=train_x.values, y=train_y.values, batch_size=32, epochs=200,
              validation_data=(valid_x.values, valid_y.values), verbose=1, callbacks=[es])


    model_file = '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/nn_' + train_air + '.model'
    model.save_weights(model_file, overwrite=True)


    # 开始进行抽取
    dense1_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_2').output)

def use_nn_to_gbdt(train_air='pm25'):
    ######################  1.书写之前的网络结构  ####################
    model = Sequential()
    model.add(Dense(activation='relu', units=2000, input_dim=3385))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=1000, name='Dense_2'))
    model.add(Dense(activation='tanh', units=1))

    ####################   加载网络模型，和加载全部数据，作前40W做初始化和提取  ###############
    f = h5py.File(
        '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/' + train_air + '.h5',
        'r')
    data = f['data'].value
    print('!')
    pm25_data_all = pd.DataFrame(data, columns=[str(i) for i in range(0, 3386)])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_all.ix[1:400000, :])
    pm25_data_all=pd.DataFrame(scaler.fit_transform(pm25_data_all.ix[1:400000, :]),columns=[str(i) for i in range(0, 3386)])

    model_file = '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/nn_' + train_air + '.model'
    model.load_weights(model_file)

    dense2_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_2').output)
    dense2_output = dense2_layer_model.predict(pm25_data_all[[str(i) for i in range(0, 3385)]])

    print(pd.DataFrame(dense2_output).head)
    #拼接标签

    #################   3.提取出来的特征作为GBDT的输入，重新训练一个模型  ########################

    dense2_output=pd.DataFrame(dense2_output,columns=[str(i) for i in range(0, 1000)])
    gbm0 = GradientBoostingRegressor(

    )

    gbm0.fit(dense2_output[[str(i) for i in range(0, 1000)]], pm25_data_all['3385'])

    model_file='/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/gbdt_nn_'+train_air+'.model'
    with open(model_file, 'wb') as fout:
        pickle.dump(gbm0, fout)
    # test_Y1 = gbm0.predict(test_X)
    # score = get_score(test_Y1, test_Y)

def gbdt_nn_predict(train_air='pm25'):
    ######################  1.书写之前的网络结构  ####################
    model = Sequential()
    model.add(Dense(activation='relu', units=2000, input_dim=3385))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=1000, name='Dense_2'))
    model.add(Dense(activation='tanh', units=1))
    ###########################################################
    f = h5py.File(
        '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/' + train_air + '.h5',
        'r')
    data = f['data'].value
    print('!')
    pm25_data_a = pd.DataFrame(data, columns=[str(i) for i in range(0, 3386)])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_a.ix[1:400000, :])
    pm25_data_all=pd.DataFrame(scaler.fit_transform(pm25_data_a.ix[500000:800000, :]),columns=[str(i) for i in range(0, 3386)])
    ###########################  用神经网络作提取  #########################
    model_file = '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/nn_' + train_air + '.model'
    model.load_weights(model_file)

    dense2_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_2').output)
    dense2_output = dense2_layer_model.predict(pm25_data_all[[str(i) for i in range(0, 3385)]])
    dense2_output = pd.DataFrame(dense2_output, columns=[str(i) for i in range(0, 1000)])

    print('dense2_output:',dense2_output)

    model_path='/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/gbdt_nn_'+train_air+'.model'
    model = pickle.load(open(model_path, 'rb'))

    the_sum_score=0
    the_sum_num=0

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_a.ix[1:400000, '3385'].reshape(-1, 1))
    for i in range(0,300):
        start_num=840*i
        end_num=(i+1)*840

        print('start_num:',start_num)
        print('end_num:',end_num)

        pm_data = dense2_output.ix[int(start_num):int(end_num), :]

        pred_PM25 = model.predict(pm_data[[str(i) for i in range(0,1000)]])


        # 真实
        #print('gggvvv:', pm25_data_all)
        pm25_data=pm25_data_all.ix[int(start_num):int(end_num), :]

        #print('ggg:',pm25_data)
        #根据自己统计的pm2.5最大最小值 作反标准化



        #print('转化前',pd.DataFrame(pred_PM25).head())
        pred_PM25=scaler.inverse_transform(pred_PM25.reshape(-1, 1))
        #print('转化后', pd.DataFrame(pred_PM25).head())
        #true=scaler.inverse_transform(pm25_data['3385'].reshape(-1, 1))


        #print('第一个：',pred_PM25)
        score = get_score(pred_PM25, pm25_data_a.ix[int(500000+start_num):int(500000+end_num), :].values)

        print(str(i)+'次，计算留出集合上损失得分：', score)
        the_sum_score=the_sum_score+score
        the_sum_num=the_sum_num+1

    print('GBDT 平均得分：',the_sum_score/the_sum_num)

def nn_predict(train_air='pm25'):

    ######################  1.书写之前的网络结构  ####################
    model = Sequential()
    model.add(Dense(activation='relu', units=2000, input_dim=3385))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(activation='relu', units=1000, name='Dense_2'))
    model.add(Dense(activation='tanh', units=1))
    ###########################################################
    model_file = '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/save_model/nn_' + train_air + '.model'
    model.load_weights(model_file)

    f = h5py.File(
        '/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/' + train_air + '.h5',
        'r')
    data = f['data'].value
    print('!')
    pm25_data_a = pd.DataFrame(data, columns=[str(i) for i in range(0, 3386)])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_a.ix[1:400000, :])

    pm25_data_all = pd.DataFrame(scaler.fit_transform(pm25_data_a.ix[500000:1000000, :]),
                                 columns=[str(i) for i in range(0, 3386)])

    data=pd.DataFrame(model.predict(pm25_data_all[[str(i) for i in range(0, 3385)]]))
    #score = model.evaluate(
    #    pm25_data_all[[str(i) for i in range(0, 3385)]], pm25_data_all['3385'], batch_size=80, verbose=1)

    the_sum_score=0
    the_sum_num=0
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(pm25_data_a.ix[1:400000, '3385'].reshape(-1, 1))
    for i in range(0,500):
        start_num=840*i
        end_num=(i+1)*840

        print('start_num:',start_num)
        print('end_num:',end_num)

        # 真实
        #print('gggvvv:', pm25_data_all)
        pm25_data=data.ix[int(start_num):int(end_num), :]

        #print('ggg:',pm25_data)
        #根据自己统计的pm2.5最大最小值 作反标准化

        print('转化前',pd.DataFrame(pm25_data).head())
        pm25_data=scaler.inverse_transform(pm25_data.values.reshape(-1, 1))
        print('转化后', pd.DataFrame(pm25_data).head())


        #print('第一个：',pm25_data)
        #print('第二个：',pm25_data_a.ix[int(500000+start_num):int(500000+end_num), '3385'])
        score = get_score(pm25_data, pm25_data_a.ix[int(500000+start_num):int(500000+end_num), '3385'].values)

        print(str(i)+'次，计算留出集合上损失得分：', score)
        the_sum_score=the_sum_score+score
        the_sum_num=the_sum_num+1

    print('GBDT 平均得分：',the_sum_score/the_sum_num)

#NN_Model()
use_nn_to_gbdt()
nn_predict()