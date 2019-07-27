from ma_util.data_process.crawl_new_data import  crawel

from ma_util.data_merge_process.merge_for_new_data.merge_grid_18_lon_lat import  get_grid_lon_lat
from ma_util.data_merge_process.data_process.merge_meo_grid import ProcessAqData
from ma_util.data_merge_process.data_process.merge_aq_grid_meo import merge_aq_grid_meo
from ma_util.data_process.data_process_together import data_process
from ma_util.data_process.csv_to_hdf5 import csv_to_h5


if __name__ == '__main__':
    crawel()
    get_grid_lon_lat()
    ProcessAqData()
    merge_aq_grid_meo()
    data_process()
    #csv_to_h5()



