import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,recall_score,precision_score
from functools import cmp_to_key
def cmp(a,b):
    if a[0]>b[0]:
        return 1
    elif a[0]<b[0]:
        return -1
    else:
        return 0
def calc_auc(y_true, y_score):
    
    return roc_auc_score(y_true, y_score)

def calc_aupr(y_true, y_score):
    
    return average_precision_score(y_true, y_score)

def calc_f1(y_true, y_score,threshold=0.5):

    y_pred = (y_score > threshold).astype(int)
    return f1_score(y_true, y_pred)

def calc_recall(y_true, y_score,threshold=0.5):

    y_pred = (y_score > threshold).astype(int)
    return recall_score(y_true, y_pred)

def calc_pre(y_true, y_score,threshold=0.5):

    y_pred = (y_score > threshold).astype(int)
    return precision_score(y_true, y_pred)

def calc_all(user_item_pair,label,pred,topk=20):
    total_f1=0
    total_pre=0
    total_recall=0
    times=0
    user=list(set(map(int,user_item_pair[:,0])))
    user.sort()
    for i in user:
        user_list=[]
        temp_pred,temp_label=pred[[user_item_pair[:,0]==i]],label[[user_item_pair[:,0]==i]]
        recall_n=0
        for i in temp_label:
            if i==1:
                recall_n+=1
        for i in range(len(temp_label)):
            user_list.append([temp_pred[i],temp_label[i]])  
        user_list.sort(reverse=True,key=cmp_to_key(cmp))
        user_list=user_list[:topk]
        predict_true=0
        
        for i in user_list:
            if i[1]==1:
                predict_true+=1
        
        user_list=np.array(user_list)
        if predict_true!=0:
            recall=predict_true/recall_n
            pre=predict_true/topk
            f1=(2*recall*pre)/(recall+pre)
        else:
            recall=0
            pre=0
            f1=0
        total_f1+=f1
        total_pre+=pre
        total_recall+=recall
        times+=1
        
    return {'recall':total_recall/times,'pre':total_pre/times,'f1':(2*total_recall/times*total_pre/times)/(total_recall/times+total_pre/times)}