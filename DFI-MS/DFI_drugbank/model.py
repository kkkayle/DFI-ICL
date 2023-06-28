from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class DFI(nn.Module):

    def __init__(self, size, embedding_size, batch_size=1024, pratio=0.5, fi_type="att"):
        super(DFI, self).__init__()
        self.embedding = FeaturesEmbedding(size, embedding_size)
        self.size = size
        self.num_field = len(size)
        self.input_size = self.num_field * embedding_size
        self.batch_size = batch_size
        self.row, self.col = list(), list()
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                self.row.append(i), self.col.append(j)
        self.dropout1 = nn.Dropout(p=pratio)
        self.dropout2 = nn.Dropout(p=pratio)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=2, dim_feedforward=1024,dropout=0.6)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.liner1 = nn.Linear(self.input_size, embedding_size)
        self.liner2 = nn.Linear(self.input_size, embedding_size)

    def forward(self, x):
        pass

    def compute_loss(self, x, alpha=0.5, beta=0.05):
        embedding = self.embedding(x)
        loss1 = self.compute_loss1(embedding)
        loss2 = self.compute_loss2(embedding)
        loss = loss2 * alpha + loss1 * beta
        return loss

    def compute_loss1(self, embedding):
        alignment_loss = torch.norm(embedding[self.row].sub(embedding[self.col]), dim=2).pow(2).mean()
        frac = torch.matmul(embedding, embedding.transpose(2, 1))
        denom = torch.matmul(torch.norm(embedding, dim=2).unsqueeze(2), torch.norm(embedding, dim=2).unsqueeze(1)) 
        separation_loss = torch.div(frac, denom + 1e-4).mean()
        return alignment_loss+separation_loss

    def compute_loss2(self, embedding):
        x_emb1, x_emb2 = self.dropout1(embedding), self.dropout2(embedding)
        x1 = self.encoder(x_emb1).view(-1, self.input_size)
        x2 = self.encoder(x_emb2).view(-1, self.input_size)
        x1 = self.liner1(x1)
        x2 = self.liner2(x2)
        loss2 = torch.norm(x1.sub(x2), dim=1).pow(2).mean()
        return loss2


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, size, embedding_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(size), embedding_size)
        self.offsets = np.array((0, *np.cumsum(size)[:-1]), dtype=np.long)
        nn.init.normal_(self.embedding.weight, std=0.01)


    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class linear_layer(torch.nn.Module):
    def __init__(self, size, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(size), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(size)[:-1]), dtype=np.long)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
    
class FM(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2  
        sum_of_square = torch.sum(x ** 2, dim=1)  
        ix = square_of_sum - sum_of_square  
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
    
class Predict_layer(DFI):
    def __init__(self, size, embedding_size, batch_size=512, pratio=0.7, fi_type="att"):
        super(Predict_layer, self).__init__(size, embedding_size, batch_size, pratio=pratio, fi_type=fi_type)
        self.liner = linear_layer(size)
        self.fm = FM(reduce_sum=True)

    def forward(self, x):
        emb_x = self.embedding(x)
        temp=self.liner(x)
        x =  temp + self.fm(emb_x)
        return x
    
class MLP(DFI):
    def __init__(self, size, embedding_size, batch_size=512, pratio=0.7, fi_type="att"):
        super(MLP, self).__init__(size, embedding_size, batch_size, pratio=pratio, fi_type=fi_type)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)