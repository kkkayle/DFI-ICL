# Importing necessary libraries
import argparse
import dataloader
import random
from tqdm import tqdm
import numpy as np
from model import DFI, Predict_layer, MultiEpochsDataLoader, MLP
import torch
from metrics import *
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# Setting up command-line arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='drugbank', help='drugbank/pubmed')
parser.add_argument('--epoch', default=270, help='')
parser.add_argument('--batch_size', default=512, help='')
parser.add_argument('--embedding_size', default=128, help='')
parser.add_argument('--drop', default=0.2, help='')
parser.add_argument('--lr', default=0.01, help='')
parser.add_argument('--weight_decay', default=1e-6, help="")
parser.add_argument('--training_interval', default=30, help="")
args = parser.parse_args()

# Function to set the random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Initialize the random seed
setup_seed(2022)

# Main execution block
if __name__ == '__main__':
    # Loading the dataset
    dataset = dataloader.Dataset('./data/' + args.dataset + '/')
    train, test = dataset.get_data()

    # Creating data loaders for training and testing
    train_loader = MultiEpochsDataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_loader = MultiEpochsDataLoader(test, batch_size=len(test), shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

    # Initializing the model and setting it to use CUDA (GPU)
    model = Predict_layer(dataset.field_dims, args.embedding_size, batch_size=args.batch_size, pratio=args.drop).cuda()
    criterion = torch.nn.BCELoss().cuda()  # Binary Cross-Entropy Loss for classification
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Adam optimizer

    # Training loop
    for epoch in tqdm(range(args.epoch - 1)):
        model.train()
        for i, (user_item_pair, label) in enumerate(train_loader):
            label = label.float().cuda()
            user_item_pair = user_item_pair.long().cuda()

            # Zero the gradients before backward pass
            optimizer.zero_grad()
            pred = torch.sigmoid(model(user_item_pair).squeeze(1))
            loss = criterion(pred, label)
            if epoch % args.training_interval == args.training_interval - 1:
                loss += model.compute_loss(user_item_pair)

            # Backpropagation
            loss.backward()
            optimizer.step()

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        for user_item_pair, label in test_loader:
            label = label.float()
            user_item_pair = user_item_pair.long().cuda()
            pred = torch.sigmoid(model(user_item_pair).squeeze(1)).cpu().detach()

            # Printing evaluation metrics
            print(f"test_AUC:{calc_auc(label, pred)} test_AUPR:{calc_aupr(label, pred)}")
            np_pred = np.array(pred)
            np_label = np.array(label)
            user_item_pair = user_item_pair.cpu()
            print(f"test_F1:{calc_f1(np_label, np_pred)}\ntest_pre:{calc_pre(np_label, np_pred)}\ntest_recall:{calc_recall(np_label, np_pred)}")
            print(f"TOP-20:{calc_all(user_item_pair, label, pred, topk=20)}")
            print(f"TOP-10:{calc_all(user_item_pair, label, pred, topk=10)}")
            print(f"TOP-5:{calc_all(user_item_pair, label, pred, topk=5)}")
