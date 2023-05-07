'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''

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

def calc_all(user_item_pair,label,pred,threshold=0.5,topk=20):
    total_f1=0
    total_pre=0
    total_recall=0
    times=0
    user=list(set(map(int,user_item_pair[:,0])))
    user.sort()
    for i in user:
        user_list=[]
        temp_pred,temp_label=pred[[user_item_pair[:,0]==i]],label[[user_item_pair[:,0]==i]]
        for i in range(len(temp_label)):
            user_list.append([temp_pred[i],temp_label[i]])  
        user_list.sort(reverse=True,key=cmp_to_key(cmp))
        user_list=user_list[:]
        predict_true=0
        for i in user_list:
            if i[1]==1:
                predict_true+=1
        
        user_list=np.array(user_list)
        f1=calc_f1(user_list[:,1],user_list[:,0])
        pre=calc_pre(user_list[:,1],user_list[:,0])
        recall=calc_recall(user_list[:,1],user_list[:,0])
        total_f1+=f1
        total_pre+=pre
        total_recall+=recall
        times+=1
        
    return total_f1/times, total_pre/times, total_recall/times



import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score, average_precision_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    testDict_neg: dict = dataset.testDict_neg
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        auc_record = []
        aupr_record = []
        label = []  
        cal_pred = []
        cnt_r = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # print(batch_users)
            allPos = dataset.getUserPosItems(batch_users)
            allNeg = dataset.getUserNegItems(batch_users)
            # print(allNeg)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)

            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)

            exclude_index_neg = []
            exclude_items_neg = []
            for range_i, items in enumerate(allNeg):
                exclude_index_neg.extend([range_i] * len(items))
                exclude_items_neg.extend(items)
            rating[exclude_index_neg, exclude_items_neg] = -(1<<10)
            # print(rating)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            #GAUC
            """
            aucs = [
                    utils.AUC(rating[i],
                              dataset, 
                              test_data) for i, test_data in enumerate(groundTrue)
                ]
            auprs = [
                    utils.AUPR(rating[i],
                              dataset, 
                              test_data) for i, test_data in enumerate(groundTrue)
                ]
            auc_record.extend(aucs)
            aupr_record.extend(auprs)
            """
            #normal auc
            cnt = []  
            for i, test_data in enumerate(groundTrue):
                r_all = np.zeros((dataset.m_items, ))
                r_all[test_data] = 1
                r = r_all[rating[i] >= 0]
                test_item_scores = rating[i][rating[i] >= 0]
                label.extend(r)
                cal_pred.extend(test_item_scores)
                cnt.append(r.shape[0])
            cnt_r.extend(cnt)

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        #assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        """
        results['auc'] = np.mean(auc_record)
        results['aupr'] = np.mean(aupr_record)
        """
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        cal_pred=np.array(cal_pred)
        
        print(f'f1:{calc_f1(label,cal_pred,0.5)} ,pre:{calc_pre(label,cal_pred,0.5)},recall:{calc_recall(label,cal_pred,0.5)}')
        print('normal auc:',roc_auc_score(label,cal_pred))
        print('normal aupr:',average_precision_score(label,cal_pred))
        
        """
        GWauroc = 0
        GWaupr = 0
        for i in range(len(cnt_r)):
            GWauroc += auc_record[i]*cnt_r[i]
            GWaupr += aupr_record[i]*cnt_r[i]
        
        s = 0
        for item in cnt_r:
            s+=item
        
        print('group and weighted auroc:',GWauroc/s)
        print('group and weighted aupr:',GWaupr/s)
        """
        return results
