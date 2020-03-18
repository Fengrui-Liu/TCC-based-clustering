import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import os
import re

def mergeTrainfile():

    mergefile = pd.DataFrame()
    path = '../Train'
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            # print(file)
            matchstr = '201.*csv'
            matchobj = re.match(matchstr, file)
            if matchobj:
                # print("!",file)
                # process_label(str(file),fea_list,tag)
                nowfile = pd.read_csv(path+'/'+str(file))
                print(nowfile.shape)
                mergefile = pd.concat([mergefile,nowfile], axis = 0)

    print("!!",mergefile.shape)

    mergefile.to_csv('../Train/train_file.csv',index = False)



if __name__ == "__main__":
    # ## A,1,disk_110667,2017-07-2l3,0

    # df = pd.read_csv('../Data/disk_sample_smart_log_201807.csv',index_col = 0)

    # result = df.loc[df.serial_number == 'disk_140241']

    # drop_list = []
    # for i in tqdm([col for col in result.columns if col not in['manufacturer','model','serial_number','dt']]):
    #     if (result[i].nunique()==1) or (result[i].isnull().sum()==result.shape[0]):
    #         drop_list.append(i)
    
    # fea_list = list(set(result.columns)-set(drop_list))
    # result = result[fea_list].sort_values('dt')


    # print(result)

    # df = pd.DataFrame(data=np.arange(20).reshape(5,4), columns=['a', 'b', 'c', 'd'])
    # print(df)
    # test = pd.DataFrame(data = np.arange(10).reshape(5,2), columns = ['a','e'])

    # df =df.merge(test['a'], on =["a"])
    
    # print(test)
    # print(df)

    # mergeTrainfile()




    # test = pd.read_csv('../Train/test.csv')
    # test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:]))
    # test['dt'] = pd.to_datetime(test['dt'])

    # print(test['serial_number'].value_counts())

    predict = pd.read_csv('../Train/predict.csv',names =['manufacturer','model','serial_number','fault_time'])

    predict = predict[predict['model']==1]
    predict.to_csv('../Train/predict_1.csv',index = 0, header = 0)