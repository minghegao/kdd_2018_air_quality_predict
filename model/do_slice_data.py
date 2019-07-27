import pandas as pd
import numpy as np
import datetime
import h5py
'''
利用经过特征工程处理好的  历史空气质量特征和天气预报特征去构建滑动数据，一行行滑动特征，和特征对应的标签分别进行存储
'''

air_data=pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/feature_eng_data/air_data_fill_wea.csv',parse_dates=['utc_time'])
weather_pred_data=pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/feature_eng_data/weather_pred.csv',parse_dates=['datetime'])
air_data=air_data[['utc_time','stationId_aq','temperature','pressure','humidity','wind_direction','wind_speed/kph','weather','NO2','CO','SO2','PM2.5','PM10','O3']]
#补充的一点预处理
#air_data=air_data.drop('Unnamed: 0',axis=1)
#air_data=air_data.drop('Unnamed: 0.1',axis=1)
weather_pred_data=weather_pred_data.interpolate()
#print('dsssssssssssdsdddddd',air_data)

def do_slice():
    #检查是否有缺失的数据
    # is_nan_air = np.isnan(air_data).any()
    # print(is_nan_air)
    # is_nan_wea = np.isnan(weather_pred_data).any()
    # print(is_nan_wea)



    ########################  第一步:获取所有的时间点  #######################
    air_min_date=np.min(air_data['utc_time'])
    air_max_date=np.max(air_data['utc_time'])
    weather_pred_min_date=np.min(weather_pred_data['datetime'])
    weather_pred_max_date=np.max(weather_pred_data['datetime'])
    print('空气质量数据长度：',air_data.shape)
    print('天气预报数据长度：',weather_pred_data.shape)
    print(air_min_date)
    print(air_max_date)
    print(weather_pred_min_date)
    print(weather_pred_max_date)

    #打算用前5天数据预测后一天     也就是144小时预测25小时的数据
    start_slide_date=air_min_date+datetime.timedelta(hours=145)
    end_slide_date=air_max_date-datetime.timedelta(hours=25)
    ########################  第二步:开始滑动把   现在滑取原始信息，使用stack堆成一行，多次把  #######################  有隐患是天气有空的
    #准备三个用于接受的dataframe:   一个是前144小时的空气质量    一个是后包括自身24小时的天气预报     一个是包括自身的3个标签值  以time和staion作为连接
    air_144_data=[]
    wea_24_data=[]
    flag_24_data=[]
    the_big_tog_data = []
    #获取每个要滑取的时间点
    '''
    思路过程是，记好每一行代表了那个站点，哪个时间，之后直接进行天气预报和气象  拼接就可以了
    '''
    num=1
    while start_slide_date<=end_slide_date:
        for station in ['donggaocun_aq', 'nansanhuan_aq', 'nongzhanguan_aq', 'pingchang_aq', 'yungang_aq',
                        'wanshouxigong_aq',
                        'aotizhongxin_aq', 'huairou_aq', 'miyunshuiku_aq', 'tongzhou_aq', 'dingling_aq', 'fangshan_aq',
                        'pinggu_aq',
                        'dongsi_aq', 'liulihe_aq', 'xizhimenbei_aq', 'badaling_aq', 'shunyi_aq', 'wanliu_aq',
                        'zhiwuyuan_aq', 'guanyuan_aq',
                        'tiantan_aq', 'yanqin_aq', 'daxing_aq', 'fengtaihuayuan_aq', 'mentougou_aq', 'yufa_aq',
                        'beibuxinqu_aq', 'dongsihuan_aq',
                        'gucheng_aq', 'qianmen_aq', 'miyun_aq', 'yizhuang_aq', 'yongledian_aq', 'yongdingmennei_aq']:

            #######################   获取前144个小时的每个站点的空气质量信息   ################################################
            the_before_start_slide_date=start_slide_date+datetime.timedelta(hours=-144)
            the_first_range_data=air_data[(air_data['stationId_aq']==station)&(air_data['utc_time']>=the_before_start_slide_date)&(air_data['utc_time']<start_slide_date)]
            the_first_range_data=the_first_range_data.drop('utc_time',axis=1)
            the_first_range_data=the_first_range_data.drop('stationId_aq', axis=1)
            first_data=list(np.hstack(the_first_range_data.values))
            #print('前144个小时对应的数据维度为：',len(first_data))
            air_144_data.append(first_data)
            #print('dddddddddddddddddddd',np.hstack(the_first_range_data.values))

            #######################   获取对应后应的后24小时的天气预报   ################################################
            the_last_slide_date = start_slide_date + datetime.timedelta(hours=23)

            #print('station:::::::::',station)
            dumname=str('station_id_'+station)
            the_secong_range_data=weather_pred_data[(weather_pred_data[dumname]==1)&(weather_pred_data['datetime']>=start_slide_date)&(weather_pred_data['datetime']<=the_last_slide_date)]
            #print('the_24_data:',the_secong_range_data)

            the_secong_range_data=the_secong_range_data.drop('datetime',axis=1)
            the_secong_range_data = the_secong_range_data.drop('Unnamed: 0', axis=1)
            second_data=list(np.hstack(the_secong_range_data.values))
            #print('后24个小时对应的数据维度为：', len(second_data))
            wea_24_data.append(second_data)
            #the_secong_range_data=the_secong_range_data.drop('stationId_aq', axis=1)
            #print('station:::::::::',b the_secong_range_data)

            #######################   获取对应的每一行对应的  未来24个小时的空气质量真实信息   ################################################
            the_last_slide_date = start_slide_date + datetime.timedelta(hours=23)
            the_first_range_data=air_data[(air_data['stationId_aq']==station)&(air_data['utc_time']>=start_slide_date)&(air_data['utc_time']<=the_last_slide_date)]
            the_first_range_data=the_first_range_data.drop('utc_time',axis=1)
            the_first_range_data=the_first_range_data.drop('stationId_aq', axis=1)
            target_data=the_first_range_data[['PM2.5','PM10','O3']]
            #print('bbbbbbbbbbbbbbbbb',target_data)
            #print('ccccccccccccccc',list(np.hstack(target_data.values)))
            third_data=list(np.hstack(target_data.values))
            #print('目标标签对应的数据维度为：', len(third_data))

            flag_24_data.append(third_data)
            #这样就记住把最后24*3 列数据中都是对应标签，可以对应位置再提取出来把
            ###################################    将三个都堆到一行里面   ####################################################################
            the_big_together=np.hstack((first_data,second_data ,third_data))
            #print('need:::::::::::',the_big_together.shape)
            the_big_tog_data.append(list(the_big_together))

            #print('现在的the_big_tog_data：',the_big_tog_data)
            #print('现在的the_big_tog_data：',pd.DataFrame(the_big_tog_data))
            print('num:',num)
            num=num+1
        start_slide_date=start_slide_date+datetime.timedelta(hours=1)
    last_big_dataset=pd.DataFrame(the_big_tog_data)
    last_big_dataset.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/feature_eng_data/do__slice_data.csv')



def do_flag_slice():
    '''
    可以根据要进行预测的标签，训练出不同的模型
    :return:
    '''
    last_big_dataset=pd.read_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/feature_eng_data/do__slice_data.csv')
    #检查下第一步处理结果是否有空缺之
    is_nan=np.isnan(last_big_dataset).any()
    print(is_nan)
    for index,i in enumerate(list(is_nan)):
        if(i==True):
            print('::::',index)

    print(last_big_dataset[last_big_dataset.isnull().values==True])


    #print(last_big_dataset.head())
    #对于最后72条数据想办法进行滑动和重组织
    the_pm25_model_use=[]
    the_pm10_model_use=[]
    the_O3_model_use=[]
    #################################### 开始对标签进行扩展  #########################
    #首先、先获取下来每次扩展都需要的部分
    the_common_data=last_big_dataset.ix[:,:-73]
    the_flag_data=last_big_dataset.ix[:,-72:]
    #print('the_common_data:',the_common_data.head())
    #print('the_flag_data:',the_flag_data.head())
    #获取发现标签位于第3384-3455列上，共72列数据

    #获取 每个标签对应的列名字
    pm25_row_name = []
    pm10_row_name = []
    O3_row_name = []
    number = 3384
    while(number <= 3455):
        pm25_row_name.append(str(number))
        pm10_row_name.append(str(number + 1))
        O3_row_name.append(str(number + 2))
        number = number + 3
    #print(the_flag_data[pm25_row_name])

    pm25_all_data=[]
    pm10_all_data=[]
    O3_all_data=[]
    ###################################  对于pm25标签，进行24次，锁定好位置，这样就将每一行都扩展成24行了，这样作为第一个模型的输入 #########################
    num=0
    print('所有的行数量： ',the_common_data.shape)

    for index,row in the_common_data.iterrows():
        #print(row)
        ############### 对于PM25开始进行拼接，对应每一行进行拼接之后都可以转成24行  把目标标签和对应预测的小时放后面###################
        for clo_num in range(0,24):
            #把基本信息   第几个小时  对应的标签信息  拼接在一起
            every_pm25=np.hstack((list(the_common_data.loc[index]), [clo_num+1], list(the_flag_data.loc[index,[pm25_row_name[clo_num]]])))
            pm25_all_data.append(every_pm25)
            num=num+1
            print('num:',num)
    print('要他：',pm25_all_data)

    f = h5py.File('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/pm25.h5', 'w')
    f['data'] = pm25_all_data
    #我们现在只做pm2.5的预测吧，数据量太大了，如果三个站点一起就得288W，其他的等会再说吧
    #pm25_dataset = pd.DataFrame(pm25_all_data)
    #pm25_dataset.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/slice_eve_data/pm25.csv')

    #想办法把时间也加进去把，发现，其实数据规律就那样，怎么过来的，自己推算下就知道了。 每个时间点对应的行数为：35*24*（时间点数量），每个35*24是在一起的
#do_flag_slice()

if __name__=='__main__':
    '''
    首先do_slice进行大的通用数据上的滑动
    在do_flag_slice上根据通用数据进行每个标签上，每个小时的滑动（是对上一步处理结果的分裂）
    '''
    #do_slice()
    do_flag_slice()