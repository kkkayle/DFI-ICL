import torch
from torch.utils import data
import argparse
import pandas as pd
from collections import defaultdict
import random
from math import ceil
class Dataset():
    def __init__(self,dataset_path,neg_ration=1):
        self.read_data(dataset_path)
        self.split_data(neg_ration)
        
    def read_data(self,dataset_path):
        train_pos_df=pd.read_csv(dataset_path+'train.txt',header=None)
        test_pos_df=pd.read_csv(dataset_path+'test.txt',header=None)
        all_pos_df=pd.concat([train_pos_df,test_pos_df],axis=0)
        self.user_size,self.item_size=self.cal_size(all_pos_df)
        self.pos_dict=defaultdict(set)
        self.neg_dict=defaultdict(set)
        all_user_set=set(range(self.item_size))
        
        for index,row in all_pos_df.iterrows():
            user=int(row[0].split()[0])
            item=set(map(eval,row[0].split()[1:]))
            self.pos_dict[user]=self.pos_dict[user].union(item)
        for key,value in self.pos_dict.items():
            self.neg_dict[key]=all_user_set-self.pos_dict[key]
            
    def split_data(self,neg_ration):
        self.train_list=None
        self.test_list=None
        
        
        for key,value in self.pos_dict.items():
            pos=torch.tensor(list(value)).unsqueeze(1)
            neg=torch.tensor(list(self.neg_dict[key])).unsqueeze(1)
            pos_user=torch.tensor([key]*len(pos)).unsqueeze(1)
            neg_user=torch.tensor([key]*len(neg)).unsqueeze(1)
            pos_label=torch.tensor([1]*len(pos)).unsqueeze(1)
            neg_label=torch.tensor([0]*len(neg)).unsqueeze(1)
            pos_pair=torch.concat([pos_user,pos,pos_label],dim=1)
            neg_pair=torch.concat([neg_user,neg,neg_label],dim=1)
            
            random_index=torch.randperm(pos_pair.shape[0])
            pos_pair=pos_pair[random_index,:]
            random_index=torch.randperm(neg_pair.shape[0])
            neg_pair=neg_pair[random_index,:]
            
            train_pos=pos_pair[:int(len(pos_pair)*0.8)]
            test_pos=pos_pair[int(len(pos_pair)*0.8):]
            
            train_neg=neg_pair[:int(len(neg_pair)*0.8)]
            test_neg=neg_pair[int(len(neg_pair)*0.8):]
            
            train=torch.concat([train_pos,train_neg],dim=0)
            test=torch.concat([test_pos,test_neg],dim=0)
            
            if self.train_list==None:
                self.train_list=train           
            else:
                self.train_list=torch.concat([self.train_list,train],dim=0)
            
            if self.test_list==None:
                self.test_list=test           
            else:
                self.test_list=torch.concat([self.test_list,test],dim=0)
            
                
                
                
        self.train=pd.DataFrame(self.train_list)
        self.test=pd.DataFrame(self.test_list)
        
        self.all_data = pd.concat([self.train, self.test])
        
        self.field_dims = [self.user_size+1,self.item_size+1]
        for i in self.all_data.columns[:2]:
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            self.test[i] = self.test[i].map(maps)
            self.train[i] = self.train[i].map(maps)
        
        self.train_loader = Dataloader(self.train)
        self.test_loader = Dataloader(self.test)
    def get_data(self):
        return self.train_loader,self.test_loader
    def out_put_data(self):
        self.train.to_csv("train.csv",index=None)
        self.test.to_csv("test.csv",index=None)
    def cal_size(self,data):
        max_user=0
        max_item=0
        for index,row in data.iterrows():
            user=int(row[0].split()[0])
            item=set(map(eval,row[0].split()[1:]))
            if user>max_user:
                max_user=user
            if max(item)>max_item:
                max_item=max(item)
        return max_user+1,max_item+1

class Dataloader():
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index].values[0:2]
        y = self.data.iloc[index].values[2]
        return x, y



"""
class Dataloader():
    def __init__(self,data,batch_size) -> None:
        self.data=torch.tensor(data)
        self.data_length=len(self.data)
        self.batch_size=batch_size
    
    
    def __getitem__(self, key):
        if key < 0 or key >= self.__len__():
            raise IndexError("Index out of range.")
        
        start_idx = key * self.batch_size
        end_idx = min((key + 1) * self.batch_size, self.data_length)
        
        return self.data[start_idx:end_idx,:2],self.data[start_idx:end_idx,2]
        
    def __len__(self):
        return ceil(self.data_length//self.batch_size)
"""