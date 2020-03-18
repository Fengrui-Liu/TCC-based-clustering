import numpy as np
import pandas as pd
import gc
# import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
# import catboost as cbt
import math
from lightgbm.sklearn import LGBMClassifier
from collections import Counter  
import time
from scipy.stats import kurtosis,iqr
from scipy import ptp
from tqdm import tqdm
import datetime
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,f1_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import warnings
import os
import joblib

if __name__ == "__main__":

    test = pd.read_csv("../Data/disk_sample_smart_log_test_a.csv")
    tag = pd.read_csv("../Data/disk_sample_fault_tag.csv")

    test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:]))
    test['dt'] = pd.to_datetime(test['dt'])
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])

    test = test.sort_values(['serial_number','dt'])
    test = test.drop_duplicates().reset_index(drop=True)
    sub = test[['manufacturer','model','serial_number','dt']]

    drop_list = []
    for i in tqdm([col for col in test.columns if col not in ['manufacturer','model']]):
        if (test[i].nunique()==1) or (test[i].isnull().sum() == test.shape[0]):
            # print(i)
            drop_list.append(i)

    fea_list = list(set(test.columns) - set(drop_list))
    # print(fea_list)
    test = test[fea_list]

    serial = pd.read_csv('../serial.csv')
    serial.columns = ['serial_number','dt_first']
    serial.dt_first = pd.to_datetime(serial.dt_first)

    tag['tag'] = tag['tag'].astype(str)
    tag = tag.groupby(['serial_number','fault_time','model'])['tag'].apply(lambda  x: '|'.join(x)).reset_index()
    tag.columns = ['serial_number','fault_time_1','model','tag']
    tag.model = tag.model.map({1:2,2:1})
    map_dict = dict(zip(tag['tag'].unique(),range(tag['tag'].nunique())))
    tag['tag'] = tag['tag'].map(map_dict).fillna(-1).astype('int32')

    feature_name = [i for i in test.columns if i not in ['dt','manufacturer']] + ['days','days_1','days_2','tag']

    ###验证
    # train_x = train_2018_4
    # del train_2018_4
    # train_x = train_x.append(train_2018_3).reset_index(drop=True)
    # del train_2018_3

    # train_x = train_x.merge(serial,how = 'left',on = 'serial_number')
    train_x = serial
    ###硬盘的使用时常
    train_x['days'] = (train_x['dt'] - train_x['dt_first']).dt.days

    train_x = train_x.merge(tag,how = 'left',on = ['serial_number','model'])
    ###当前时间与另一个model故障的时间差，
    train_x['days_1'] = (train_x['dt'] - train_x['fault_time_1']).dt.days
    train_x.loc[train_x.days_1 <= 0,'tag'] = None
    train_x.loc[train_x.days_1 <= 0,'days_1'] = None

    ###同一硬盘第一次使用到开始故障的时间
    train_x['days_2'] = (train_x['fault_time_1'] - train_x['dt_first']).dt.days
    train_x.loc[train_x.fault_time_1 >= train_x.dt,'days_2'] = None
    

    train_x['serial_number'] = train_x['serial_number'].apply(lambda x:int(x.split('_')[1]))
    train_y = train_x.label.values
    train_x = train_x[feature_name]
    gc.collect()

    val_x = train_2018_6
    del train_2018_6

    val_x = val_x.merge(serial,how = 'left',on = 'serial_number')
    val_x['days'] = (val_x['dt'] - val_x['dt_first']).dt.days


    val_x = val_x.merge(tag,how = 'left',on = ['serial_number','model'])
    val_x['days_1'] = (val_x['dt'] - val_x['fault_time_1']).dt.days
    val_x.loc[val_x.days_1 <= 0,'tag'] = None
    val_x.loc[val_x.days_1 <= 0,'days_1'] = None
    val_x['days_2'] = (val_x['fault_time_1'] - val_x['dt_first']).dt.days
    val_x.loc[val_x.fault_time_1 >= val_x.dt,'days_2'] = None

    val_x['serial_number'] = val_x['serial_number'].apply(lambda x:int(x.split('_')[1]))
    val_y = val_x.label.values
    val_x = val_x[feature_name]
    gc.collect()


    gc.collect()

    clf = LGBMClassifier(   
        learning_rate=0.001,
        n_estimators=10000,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2019,
        is_unbalenced = 'True',
        metric=None
    )
    print('************** training **************')
    print(train_x.shape,val_x.shape)
    clf.fit(
        train_x, train_y,
        eval_set=[(val_x, val_y)],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=100
    )

    ##根据验证的轮数训练
    train_df = train_2018_6.append(train_2018_5).reset_index(drop=True)
    del train_2018_6
    del train_2018_5
    train_df = train_df.append(train_2018_4).reset_index(drop=True)
    del train_2018_4


    train_df = train_df.merge(serial,how = 'left',on = 'serial_number')
    train_df['days'] = (train_df['dt'] - train_df['dt_first']).dt.days

    train_df = train_df.merge(tag,how = 'left',on = ['serial_number','model'])
    train_df['days_1'] = (train_df['dt'] - train_df['fault_time_1']).dt.days
    train_df.loc[train_df.days_1 <= 0,'tag'] = None
    train_df.loc[train_df.days_1 <= 0,'days_1'] = None
    train_df['days_2'] = (train_df['fault_time_1'] - train_df['dt_first']).dt.days
    train_df.loc[train_df.fault_time_1 >= train_df.dt,'days_2'] = None


    train_df['serial_number'] = train_df['serial_number'].apply(lambda x:int(x.split('_')[1]))
    labels = train_df.label.values
    train_df = train_df[feature_name]
    gc.collect()


    test = test.merge(serial,how = 'left',on = 'serial_number')
    test['days'] = (test['dt'] - test['dt_first']).dt.days

    test = test.merge(tag,how = 'left',on = ['serial_number','model'])
    test['days_1'] = (test['dt'] - test['fault_time_1']).dt.days
    test.loc[test.days_1 <= 0,'tag'] = None
    test.loc[test.days_1 <= 0,'days_1'] = None
    test['days_2'] = (test['fault_time_1'] - test['dt_first']).dt.days
    test.loc[test.fault_time_1 >= test.dt,'days_2'] = None


    test['serial_number'] = test['serial_number'].apply(lambda x:int(x.split('_')[1]))
    test_x = test[feature_name]
    del test


    gc.collect()
    clf = LGBMClassifier(
        learning_rate=0.001,
        n_estimators=100,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2019,
    #     is_unbalenced = 'True',
        metric=None
    )
    print('************** training **************')
    print(train_df.shape,test_x.shape)
    clf.fit(
        train_df, labels,
        eval_set=[(train_df, labels)],
        eval_metric='auc',
        early_stopping_rounds=10,
        verbose=10
    )

    sub['p'] = clf.predict_proba(test_x)[:,1]
    sub['label'] = sub['p'].rank()
    sub['label']= (sub['label']>=sub.shape[0] * 0.996).astype(int)
    submit = sub.loc[sub.label == 1]
    ###这里我取故障率最大的一天进行提交，线上测了几个阈值，100个左右好像比较好。。。。
    submit = submit.sort_values('p',ascending=False)
    submit = submit.drop_duplicates(['serial_number','model'])
    submit[['manufacturer','model','serial_number','dt']].to_csv("../sub.csv",index=False,header = None)
    submit.shape