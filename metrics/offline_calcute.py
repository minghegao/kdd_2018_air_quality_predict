# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:46:55 2018
@author: ZHILANGTAOSHA
"""
import numpy as np
import pandas as pd
from requests.exceptions import RequestException
import requests

stat_latlon = pd.read_csv('/home/fly/PycharmProjects/version2-baseline-4-28/DeepST-KDD_for_predict/for_submit_data/stat1.csv', header=None)
stat_latlon = stat_latlon[[0]]
new_list=[]
for index,row in stat_latlon.iterrows():
    new_list.append(row[0].split(' '))
stat_latlon=pd.DataFrame(new_list,columns=['dq', 'stationId', 'lat', 'lon'])
print(stat_latlon)

stat_latlon.columns = ['dq', 'stationId', 'lat', 'lon']
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
}
cw = {'wanshouxig_aq': 'wanshouxigong_aq', 'aotizhongx_aq': 'aotizhongxin_aq', 'nongzhangu_aq': 'nongzhanguan_aq',
      'fengtaihua_aq': 'fengtaihuayuan_aq',
      'miyunshuik_aq': 'miyunshuiku_aq', 'yongdingme_aq': 'yongdingmennei_aq', 'xizhimenbe_aq': 'xizhimenbei_aq'}
gz = {value: key for key, value in cw.items()}


def colchuli(x):
    if x == 'time': return 'utc_time'
    if x == 'station_id': return 'stationId'
    x = str(x).replace('_Concentration', '').replace('.', '').upper()
    if x == 'PM25':
        return 'PM2.5'
    else:
        return x


def chulifloat(x):
    try:
        return float(x)
    except:
        return np.NaN


def get_xianshang_true(times, city):
    time_start = (pd.to_datetime(times) - pd.DateOffset(days=1)).strftime('%Y-%m-%d-%H')
    time_end = (pd.to_datetime(times) + pd.DateOffset(days=2)).strftime('%Y-%m-%d-%H')
    url = 'https://biendata.com/competition/airquality/%s/%s/%s/2k0d1d8' % (city, time_start, time_end)
    print(url)
    zt = 0
    while zt == 0:
        try:
            respones = requests.get(url, headers=headers)
            zt = 1
            print(respones.status_code)
        except RequestException:
            print('Error')
            zt = 0
    df = pd.DataFrame([i.decode('utf8').split(',') for i in respones.content.splitlines()])
    df.columns = df.loc[0]
    df = df.shift(-1, axis=0).dropna()
    df.columns = df.columns.map(colchuli)
    df.utc_time = pd.to_datetime(df.utc_time)
    floatcl = [i for i in df.columns if i not in ['stationId', 'utc_time']]
    df.loc[:, floatcl] = df.loc[:, floatcl].applymap(chulifloat)
    return df


def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b != 0, casting='unsafe'))


def get_true(times, smaplev='old'):
    xlj = get_xianshang_true(times, 'bj')
    bjtrue = xlj[(xlj.utc_time >= times) & (xlj.utc_time < pd.to_datetime(times) + pd.DateOffset(days=2))]
    xlj = get_xianshang_true(times, 'ld')
    ldtrue = xlj[(xlj.utc_time >= times) & (xlj.utc_time < pd.to_datetime(times) + pd.DateOffset(days=2))]
    ldtrue['O3'] = 0
    true = pd.concat([bjtrue[['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']]
                         , ldtrue[['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']]])
    true['hour'] = true.utc_time.dt.hour
    true['hour'] = true.apply(
        lambda x: x['hour'] + 24 if x['utc_time'] >= pd.to_datetime(times) + pd.DateOffset(days=1) else x['hour'],
        axis=1)
    if smaplev == 'old':
        true['test_id'] = true.stationId.map(lambda x: gz[x] if x in gz.keys() else x) + '#' + (true.hour).astype(str)
    else:
        true['test_id'] = true.stationId + '#' + (true.hour).astype(str)
    true = true[true.stationId.isin(stat_latlon.stationId)]
    return true


def smape_pf_sub(true, sub2):
    smapedf = []
    truez = []
    predz = []
    for label in ['PM2.5', 'PM10', 'O3']:
        ck = pd.merge(true[['test_id', label]], sub2[['test_id', label]], on='test_id', how='left')
        ck = ck[ck['%s_x' % label].notnull()]
        ck = ck[(ck['%s_x' % label] != 0) & (ck['%s_y' % label] != 0)]
        smapedf.append(smape(ck['%s_x' % label], ck['%s_y' % label]))
        predz.extend(ck['%s_x' % label].tolist())
        truez.extend(ck['%s_y' % label].tolist())
        print('%s:' % label, smape(ck['%s_x' % label], ck['%s_y' % label]))
    print(smape(np.array(truez), np.array(predz)))
    return smapedf


true = get_true('2018-05-09 00:00:00')
true = true.dropna()
sub2 = pd.read_csv('/home/fly/Desktop/my_submissioin.csv')
true1 = true[true['O3'] == 0]
true2 = true[true['O3'] != 0]
jg2 = smape_pf_sub(true1, sub2)
jg1 = smape_pf_sub(true2, sub2)
print('the  last score',(np.mean(jg1) + np.mean(jg2[0:2])) / 2)