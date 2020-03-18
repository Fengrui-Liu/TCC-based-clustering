import tensorflow.compat.v2 as tf
import pywt
import math
import numpy as np
class Point(object):
    def __init__(self,timestamp,value):
        self.timestamp = timestamp
        self.value = value
        self.front_grad = float("inf")
        self.back_grad = float("inf")
    def set_grad(self, f_grad,b_grad):
        self.front_grad = f_grad
        self.back_grad = b_grad


def cal_grad(seq):

    seq_len = len(seq)
    grad_sequence = []
    if seq_len < 3:
        # print("Sequence is too short")
        return None
    
    sum_result = 0
    grad_sequence = []
    for i in range(1,seq_len-1):
        # print(seq[i].value)
        # print(seq[i-1].value)
        # approx_grad = 0.5*((seq[i].value - seq[i-1].value)/(seq[i].timestamp-seq[i-1].timestamp) + (seq[i+1].value-seq[i].value)/(seq[i+1].timestamp-seq[i].timestamp))
        # v  = (seq[i].value - seq[i-1].value)/(seq[i].timestamp-seq[i-1].timestamp)
        # print(v)
        # f_grad =np.divide( np.arctan(v)*200, np.pi)
        # print(f_grad)
        # v = (seq[i+1].value - seq[i].value)/(seq[i+1].timestamp-seq[i].timestamp)
        # b_grad = np.divide( np.arctan(v)*200, np.pi)
        # print(b_grad)
        # seq[i].set_grad(f_grad,b_grad)
        
        # f_grad = f_grad.tolist()
        # f_grad.extend(seq[i].value.tolist())
        # f_grad.extend(b_grad.tolist())
        # print(f_grad)
        # grad_sequence.append(f_grad)

        grad_sequence.append(seq[i].value.tolist())
        # print(grad_sequence)
        # result = np.hstack((f_grad,seq[i].value,b_grad))
        
        # grad_sequence = np.vstack((grad_sequence,result))
        # print(grad_sequence.shape)
        # print(result)
        # grad_sequence.append([f_grad*900000,b_grad*900000,seq[i].value*1000])
        
    return grad_sequence
        









def wavelet_denoising(data):
    # 小波函数取db4
    db4 = pywt.Wavelet('db4')
    if type(data) is not None:
        # 分解
        coeffs = pywt.wavedec(data, db4,2)
        # 高频系数置零
        # print(len(coeffs))
        # coeffs[len(coeffs)-1] *= 0
        # coeffs[len(coeffs)-2] *= 0
        # coeffs[len(coeffs)-3] *= 0
        # coeffs[len(coeffs)-4] *= 0
        # coeffs[len(coeffs)-5] *= 0
        # coeffs[len(coeffs)-6] *= 0
        # coeffs[len(coeffs)-7] *= 0
        # 重构
        meta = pywt.waverec(coeffs, db4)
        return coeffs,meta


def cal_seq(data):
    data_series = []
    data_grad_seq = []
    for i,v in enumerate(data):
        data_series.append((i,np.array(v)))

    data_seq = [Point(x[0],x[1]) for x in data_series]
    data_grad_seq = cal_grad(data_seq)

    return data_grad_seq


def binarySearch (arr, l, r, x): 
  
    # 基本判断
    if r >= l: 
  
        mid = int(l + (r - l)/2)
        # print(mid)
        
        # 元素整好的中间位置
        if  arr[mid] < x < arr[mid+1]: 
            return mid + 1 
          
        # 元素小于中间位置的元素，只需要再比较左边的元素
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
  
        # 元素大于中间位置的元素，只需要再比较右边的元素
        else: 
            return binarySearch(arr, mid+1, r, x) 
  
    else: 
        # 不存在
        return -1


# def cal_longest_subsequence(softmaxed_logits):
    
#     int_logits = tf.dtypes.cast(tf.round(softmaxed_logits),dtype=tf.int32)
#     index_tensor = tf.range(softmaxed_logits.shape[1],dtype=tf.int32)
#     t_index = tf.reshape(index_tensor,[softmaxed_logits.shape[1],1])
#     new_seq =tf.transpose( tf.matmul(int_logits,t_index))[0].numpy().tolist()

#     # new_seq = [3,2,4,5,6,5,5,6,7]
#     # print(new_seq)
#     subseq = []
#     indexseq = []
#     for i in range(len(new_seq)):
#         if i==0:
#             subseq.append(new_seq[i])
#             indexseq.append(i)
#         else:
#             if new_seq[i] > subseq[-1]:
#                 subseq.append(new_seq[i])
#                 indexseq.append(i)
#             elif new_seq[i] < subseq[0]:
#                 subseq[0] = new_seq[i]
#                 indexseq[0] = i
#             else:
#                 index = binarySearch(subseq,0,len(subseq)-1,new_seq[i])
#                 if index != -1:
#                     subseq[index] = new_seq[i]
#                     indexseq[index] = i
#     # print(subseq)
#     # print(indexseq)

#     subseq_tensor = tf.reshape(subseq,[1,-1])
#     index_tensor = tf.reshape(indexseq,[1,-1])
#     # print(subseq_tensor,index_tensor)
#     te = tf.subtract(subseq_tensor,index_tensor)
#     # print(te)
#     minus_result = tf.square(tf.subtract(subseq_tensor,index_tensor))
#     one_tensor = tf.ones([1,len(subseq)],tf.int32)

#     result = tf.divide(one_tensor, tf.add(one_tensor,minus_result))

#     # return tf.reduce_sum(result)
#     return len(subseq)


def cal_longest_subsequence(softmaxed_logits):
    
    int_logits = tf.dtypes.cast(tf.round(softmaxed_logits),dtype=tf.int32)
    index_tensor = tf.range(softmaxed_logits.shape[1],dtype=tf.int32)
    t_index = tf.reshape(index_tensor,[softmaxed_logits.shape[1],1])
    new_seq =tf.transpose( tf.matmul(int_logits,t_index))[0].numpy().tolist()

    # new_seq = [3,2,4,5,6,5,5,6,7]
    # print(new_seq)
    subseq = []
    indexseq = []
    for i in range(len(new_seq)):
        if i==0:
            subseq.append(new_seq[i])
            indexseq.append(i)
        else:
            if new_seq[i] > subseq[-1]:
                subseq.append(new_seq[i])
                indexseq.append(i)
            elif new_seq[i] < subseq[0]:
                subseq[0] = new_seq[i]
                indexseq[0] = i
            else:
                index = binarySearch(subseq,0,len(subseq)-1,new_seq[i])
                if index != -1:
                    subseq[index] = new_seq[i]
                    indexseq[index] = i
    # print(subseq)
    # print(indexseq)

    subseq_tensor = tf.reshape(subseq,[1,-1])
    index_tensor = tf.reshape(indexseq,[1,-1])
    # print(subseq_tensor,index_tensor)
    te = tf.subtract(subseq_tensor,index_tensor)
    # print(te)
    minus_result = tf.square(tf.subtract(subseq_tensor,index_tensor))
    one_tensor = tf.ones([1,len(subseq)],tf.int32)

    result = tf.divide(one_tensor, tf.add(one_tensor,minus_result))

    # return tf.reduce_sum(result)
    return subseq