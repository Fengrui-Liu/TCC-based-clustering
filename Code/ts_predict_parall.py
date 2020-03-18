from tcc.deterministic_alignment import align_pair_of_sequences
from tcc.alignment import compute_alignment_loss
from cal import *
import pywt
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
from joblib import Parallel, delayed
import multiprocessing
# import statsmodels.api as sm

# pd.set_option('display.max_rows', 1500)
# pd.set_option('display.max_columns', 1100)

tqdm.pandas()
def applyParallel(dfGrouped,func,train_raw_group):
    
    retLst  = Parallel(n_jobs = multiprocessing.cpu_count())(delayed(func)(group,train_raw_group) for name, group in  dfGrouped)
    print("finish")
    return pd.concat(retLst,ignore_index=True)

def cal_logits_parall(group_test,train_raw_group):
    predict_result = pd.DataFrame(columns=['manufacturer','model','serial_number','fault_time'])

    serial_numeber_now = np.array(group_test.serial_number).tolist()[0]
    model_now = np.array(group_test.model).tolist()[0]
    manufa_now = np.array(group_test.manufacturer).tolist()[0]
    # print(serial_numeber_now,model_now,manufa_now)

    group_test = group_test.sort_values('dt')
    series_group = group_test[series_list]
    data_now =np.array(series_group).tolist()
    dt_now = group_test.iloc[-1]
    # print(dt_now)
    
    test_seq = cal_seq(data_now)
    
    vote_seq = {}

    if len(data_now)<5:
        # continue
        return predict_result
    # print(test_seq)
    for name, group in train_raw_group:
        # group = group.sort_values('dt')
        train_group = group[series_list]
        data_train = np.array(train_group).tolist()
        tag_train =np.array(group.tag).tolist()[0]
        # print(tag_train)
        # print(data_train)
        # print(data_train)
        train_seq = cal_seq(data_train)
        
        if train_seq != None :
            logits = align_pair_of_sequences(train_seq, test_seq,'l2',0.1)
            # print(logits)
            # print(train_seq)
            
            raw_len = logits.shape[0]
            
            sub_seq = cal_longest_subsequence(logits)
            len_sub_seq = len(sub_seq)
            
            frac = round(len_sub_seq/len(test_seq),2)
            if frac >= 0.4:
                # print(len(train_seq),len(test_seq))
                # print(logits.shape[0])
                # print(len_sub_seq)
                # print(sub_seq[-1])
                # print(sub_seq)
                gap = raw_len - sub_seq[-1]
                if gap >=0:
                    
                    if gap in vote_seq:
                        vote_seq[gap]+= frac
                    else:

                        vote_seq[gap] = frac
        # break
        
    if len(vote_seq) !=0:

        dt_gap = max(vote_seq.keys(), key=(lambda x:vote_seq[x]))
        dt_fault = dt_now['dt'] + datetime.timedelta(days=gap)
        # dt_fault = dt_fault.ix[0]
        # print(dt_fault)
        predict_result = predict_result.append([{'manufacturer':manufa_now,'model':model_now,'serial_number':serial_numeber_now,'fault_time': dt_fault}])
    
    print(predict_result)
    return predict_result
    # count+=1
    # print(count,dt_fault)

if __name__ == "__main__":

    series_list = ['smart_198_normalized', 'smart_184_normalized', 'smart_195_normalized', 'smart_194_normalized', 'smart_193_normalized', 'smart_192_normalized', 'smart_1_normalized', 'smart_4_normalized', 'smart_197_normalized', 'smart_7_normalized', 'smart_12_normalized', 'smart_190_normalized', 'smart_5_normalized', 'smart_3_normalized', 'smart_9_normalized', 'smart_189_normalized', 'smart_187_normalized', 'smart_188_normalized']

    test = pd.read_csv('../Train/test.csv')
    test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:]))
    test['dt'] = pd.to_datetime(test['dt'])
    # print(test.shape)

    train = pd.read_csv('../Train/train_file.csv')
    train['dt'] = train['dt'].apply(lambda x:''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:]))
    train['dt'] = pd.to_datetime(train['dt']).dt.date


    
    count = 0


    train_raw_group = train.groupby(['serial_number'])


    predict_result = applyParallel(test.groupby(['serial_number','model']),cal_logits_parall,train_raw_group)

    # for name_train, group_test in test.groupby(['serial_number','model']):
        


    predict_result.to_csv('../Train/predict_lite.csv', index = 0, header = 0)



    



    
    
    



    



    