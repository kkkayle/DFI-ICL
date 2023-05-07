import pandas as pd

train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

train_pos=train_df[train_df['2']==1]
train_neg=train_df[train_df['2']==0]
test_pos=test_df[test_df['2']==1]
test_neg=test_df[test_df['2']==0]

with open('train.txt','w') as train:
    user=set()
    begin=True
    for index,row in train_pos.iterrows():
        if row[0] not in user:
            if begin==True:
                train.write(f"{row[0]} ")
                begin=False
            else:
                train.write(f"\n{row[0]} ")
            user.add(row[0])
        train.write(f"{row[1]} ")

with open('train_neg.txt','w') as train_n:
    user=set()
    begin=True
    for index,row in train_neg.iterrows():
        if row[0] not in user:
            if begin==True:
                train_n.write(f"{row[0]} ")
                begin=False
            else:
                train_n.write(f"\n{row[0]} ")
            user.add(row[0])
        train_n.write(f"{row[1]} ")
        
with open('test.txt','w') as test:
    user=set()
    begin=True
    for index,row in test_pos.iterrows():
        if row[0] not in user:
            if begin==True:
                test.write(f"{row[0]} ")
                begin=False
            else:
                test.write(f"\n{row[0]} ")
            user.add(row[0])
        test.write(f"{row[1]} ")

