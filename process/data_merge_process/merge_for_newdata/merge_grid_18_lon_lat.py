import pandas as pd
#用于为每个网格获取经纬度(step1)
def get_grid_lon_lat():
    #因为文件格式存在小问题，直接读取会出现“DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.”
    #错误，解决方法为（1）显示指出dtype={'id':int }或者low_memor=False
    grid_18=pd.read_csv("/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/origion_data/gird_2018_3_31.csv",low_memory=False)
    grid_17=pd.read_csv("/home/fly/PycharmProjects/gmh/18_data_merge_process/data/Beijing_historical_meo_grid.csv",dtype={'stationName':str})
    lon_lat=pd.DataFrame(grid_17,columns=['stationName','longitude','latitude'])
    #此步骤很重要，否则会产生memory错误
    dr_lon_lat=lon_lat.drop_duplicates(['longitude', 'latitude'])
    df_grid_18=pd.DataFrame(grid_18)
   # print(df_grid_18.head())
   # print (lon_lat.head())
   # rename_df_grid_18=pd.DataFrame(df_grid_18.rename({'station_id':'stationName'},inplace=True))   inner是交集方式的连接
    meo=pd.merge( df_grid_18,dr_lon_lat,on ='stationName',how='inner')
    meo.to_csv('/home/fly/PycharmProjects/our_competition_project/KDD_air_pollution/data/process_data/grid_18_meo.csv',
                       index=False)
    print(meo.head())
get_grid_lon_lat()