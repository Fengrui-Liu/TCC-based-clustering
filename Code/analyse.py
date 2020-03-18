import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder



from tqdm import tqdm
import joblib
import re
import os

# def data_clean(in_file,out_file):
#     df = pd.read_csv(in_file, index_col = 0)
    
#     rows_num = df.shape[0]
#     df = df.dropna(axis=1, how='all',thresh=rows_num/2)
#     cols_num = df.shape[1]
#     df = df.dropna(axis=0, how='all', thresh = cols_num/2)
#     rows_num = df.shape[0]
#     cols_num = df.shape[1]
#     # print(df)
#     na_rows, na_cols = np.where(pd.isna(df))
#     print(na_rows,na_cols)
#     for item in range(len(na_rows)):
#         r = na_rows[item]
#         c = na_cols[item]
#         serial = df.iloc[r].serial_number
#         df.iloc[r,c] =  df.loc[df.serial_number == serial].iloc[:,c].median()
        

#     df.to_csv(out_file, index = 0)



# if __name__ == "__main__":
    
#     # in_file = "../Data/disk_sample_smart_log_201707.csv"
#     # out_file = "../Data/disk_sample_smart_log_201707_clean.csv"

#     in_file = "../Data/disk_sample_smart_log_201708.csv"
#     out_file = "../Data/disk_sample_smart_log_201708_clean.csv"

#     data_clean(in_file,out_file)
    
#     df = pd.read_csv("../Data/disk_sample_smart_log_201707_clean.csv")

#     print(df.isnull().sum())
#     # print(rows_num,cols_num)


def get_label(df, fea_list, tag):
    df = df[fea_list]
    df['dt'] = df['dt'].apply(lambda x:''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:]))
    df['dt'] = pd.to_datetime(df['dt'])
    print('merge start')
    # 通过serial_number 和model 将训练数据与标签数据合并
    df = df.merge(tag[['serial_number','model','fault_time','tag']],how='left',on=['serial_number','model'])
    print('merge finish')
    df['diff_day'] = (df['fault_time']-df['dt']).dt.days
    df['label'] = 0
    df.loc[(df['diff_day']>=0)&(df['diff_day']<=30),'label'] = 1

    # df = df.groupby(['serial_number','model'])
    print(df)
    return df

def process_label(file,fea_list,tag):
    train_file = pd.read_csv('../Data/' + file, chunksize= 10000)
    print('!!!!!!')
    train_file = get_label(train_file,fea_list,tag)
    # joblib.dump(train_file,'../JoblibData/'+ file[:-3] +'jl.z')
    return

if __name__ == "__main__":

    # 处理 test， tag的日期格式
    test = pd.read_csv("../Data/disk_sample_smart_log_test_a.csv")
    tag = pd.read_csv("../Data/disk_sample_fault_tag.csv")
    test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:]))
    test['dt'] = pd.to_datetime(test['dt'])
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])

    ## 存在一种硬盘对应多种故障

    tag['tag'] = tag['tag'].astype(str)
    tag = tag.groupby(['serial_number','fault_time','model'])['tag'].apply(lambda x:'|'.join(x.sort_values())).reset_index()

    # print(tag.tag.value_counts())

    test = test.sort_values(['serial_number','dt'])
    test = test.drop_duplicates().reset_index(drop=True)

    # 选取测试集中无效的列，将其丢掉
    drop_list = []

    for i in tqdm([col for col in test.columns if col not in ['manufacturer','model']]):
        if (test[i].nunique()==1) or (test[i].isnull().sum() == test.shape[0]):
            # print(i)
            drop_list.append(i)


    fea_list = list(set(test.columns) - set(drop_list))
    # print(fea_list)
    test = test[fea_list]

    print(tag)

    # path = '../Data'
    # for dirpath, dirnames, filenames in os.walk(path):
    #     for file in filenames:
    #         matchstr = 'disk_sample_smart_log_2017.*\.csv'
    #         matchobj = re.match(matchstr, file)
    #         if matchobj:
    #             print(file)
    #             process_label(str(file),fea_list,tag)

    # serial = pd.DataFrame()
    # path = '../JoblibData'
    # for dirpath, dirnames, filenames in os.walk(path):
    #     for file in filenames:
    #         train_file = joblib.load('../JoblibData/'+str(file))[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
    #         serial = pd.concat((serial,train_file),axis=0)
            

    # serial = serial.sort_values('dt').drop_duplicates('serial_number').reset_index(drop=True)
    # serial.columns=['serial_number','dt_first']
    # serial.dt_first = pd.to_datetime(serial.dt_first)
    # serial.to_csv('../serial.csv',index=False)

    
    
    

