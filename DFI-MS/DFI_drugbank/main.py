import argparse
import dataloader
import random
from  tqdm import  tqdm
import numpy as np
from model import DFI,Predict_layer,MultiEpochsDataLoader,MLP
import torch
from metrics import *
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',default='drugbank',help='drugbank/pubmed')
parser.add_argument('--epoch',default=270,help='')
parser.add_argument('--batch_size',default=512,help='')
parser.add_argument('--embedding_size',default=128,help='')
parser.add_argument('--drop',default=0.2,help='')
parser.add_argument('--lr',default=0.01,help='')
parser.add_argument('--weight_decay', default=1e-6, help="")
parser.add_argument('--training_interval',default=30, help="")
args=parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2022)

if __name__=='__main__':
    dataset=dataloader.Dataset('./data/'+args.dataset+'/')
    train,test=dataset.get_data()
    #dataset.out_put_data()
    train_loader = MultiEpochsDataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=8, pin_memory=True)
    test_loader = MultiEpochsDataLoader(test, batch_size=len(test),shuffle=False, drop_last=False,num_workers=8, pin_memory=True)

    model=Predict_layer(dataset.field_dims,args.embedding_size,batch_size=args.batch_size,pratio=args.drop).cuda()
    #model=MLP(dataset.field_dims,args.embedding_size,batch_size=args.batch_size,pratio=args.drop).cuda()
    criterion=torch.nn.BCELoss().cuda()
    flag=False
    optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for epoch in tqdm(range(args.epoch-1)):
        model.train()
        #pred_list=[]
        #label_list=[]
        for i, (user_item_pair, label) in enumerate(train_loader):
            label=label.float().cuda()
            user_item_pair = user_item_pair.long().cuda()

            optimizer.zero_grad()
            pred=torch.sigmoid(model(user_item_pair).squeeze(1))
            loss=criterion(pred, label)
            if epoch%args.training_interval==args.training_interval-1:
                loss+=model.compute_loss(user_item_pair)
            #print(f'bce_loss:{loss}')

            loss.backward()
            optimizer.step()
            #pred_list.extend(pred.tolist())
            #label_list.extend(label.tolist())
    model.eval()
    with torch.no_grad():
        for user_item_pair,label in test_loader:
            label=label.float()
            user_item_pair = user_item_pair.long().cuda()
            pred=torch.sigmoid(model(user_item_pair).squeeze(1)).cpu().detach()
            print(f"test_AUC:{calc_auc(label,pred)} test_AUPR:{calc_aupr(label,pred)}")

            np_pred=np.array(pred)
            np_label=np.array(label)
            user_item_pair=user_item_pair.cpu()
            print(f"test_F1:{calc_f1(np_label,np_pred)}\ntest_pre:{calc_pre(np_label,np_pred)}\ntest_recall:{calc_recall(np_label,np_pred)}")
            
            print(f"TOP-20:{calc_all(user_item_pair,label,pred,topk=20)}")
            print(f"TOP-10:{calc_all(user_item_pair, label, pred, topk=10)}")
            print(f"TOP-5:{calc_all(user_item_pair, label, pred, topk=5)}")
            


