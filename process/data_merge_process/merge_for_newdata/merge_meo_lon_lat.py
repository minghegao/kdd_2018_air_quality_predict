import pandas as pd
#用于为每个网格获取经纬度(step1)
def get_meo_lon_lat():
    #因为文件格式存在小问题，直接读取会出现“DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.”
    #错误，解决方法为（1）显示指出dtype={'id':int }或者low_memor=False
    meo_18=pd.read_csv("/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/origion_data/18weather_2018_3_31.csv",low_memory=False)
    meo_17=pd.read_csv("/home/fly/PycharmProjects/gmh/data_merge_process/data/beijing_17_18_meo.csv")
    lon_lat=pd.DataFrame(meo_17,columns=['station_id','longitude','latitude'])
    #此步骤很重要，否则会产生memory错误
    dr_lon_lat=lon_lat.drop_duplicates(['longitude', 'latitude'])
    df_meo_18=pd.DataFrame(meo_18)
   # print(df_grid_18.head())
   # print (lon_lat.head())
   # rename_df_grid_18=pd.DataFrame(df_grid_18.rename({'station_id':'stationName'},inplace=True))
    meo=pd.merge( df_meo_18,dr_lon_lat,on ='station_id',how='inner')
    meo.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/18_meo_station.csv',
                       index=False)
    print(meo.head())


get_meo_lon_lat()