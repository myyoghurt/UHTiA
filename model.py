#!/PycharmProjects/env python
#-*- coding:utf-8 -*-
#!/PycharmProjects/env python
#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from attention_model import Sublayers,PositionalEncoding,Attention,LayerNorm
import copy
import collections
import numpy as np
import math
import logging

class LST(nn.Module):
    def __init__(self, model_args,device):
        super(LST, self).__init__()
        self.args = model_args
        self.device = device
        d_model = self.args.d_model
        dropout = self.args.dropout
        self.padding=0
        # user,item and timedlt embeddings,共num_items行的look-up表，每个item为d_model维特征
        self.item_embeddings = nn.Embedding(self.args.num_items, d_model, padding_idx=0).to(self.device)#padding_idx，则表示这个位置处的向量值都是零向量。
        self.user_embeddings = nn.Embedding(self.args.num_users, d_model, padding_idx=0).to(self.device)
        self.timedlt_embeddings = nn.Embedding(self.args.Threshold_timedlt, d_model).to(self.device)

        self.intra_attention = Attention(d_model, dropout,self.device)
        self.pos_embedding = PositionalEncoding(d_model,dropout).to(self.device)
        self.sublayers =Sublayers(d_model, self.args.h, self.args.d_ff, dropout,self.device)#FFN d_ff:内层维度
        self.Norm=LayerNorm(d_model,self.device)

        self.gate_Wl = Variable(torch.zeros(d_model, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_Wl = torch.nn.init.xavier_uniform_(self.gate_Wl)
        self.gate_Ws = Variable(torch.zeros(d_model, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_Ws = torch.nn.init.xavier_uniform_(self.gate_Ws)
        self.gate_Wt = Variable(torch.zeros(d_model, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_Wt = torch.nn.init.xavier_uniform_(self.gate_Wt)
        self.gate_bias = Variable(torch.zeros(1, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_bias = torch.nn.init.xavier_uniform_(self.gate_bias)

    def forward(self,batch_us,batch_sessions, batch_timedlt, batch_to_predict):
        seq=torch.from_numpy(np.array(batch_sessions)).type(torch.LongTensor).to(self.device)
        E_embs = self.item_embeddings(Variable(seq, requires_grad=False))  # [batch_size , sessions_len , items , item_d_model]
        U_embs = self.user_embeddings(Variable(torch.from_numpy(np.array(batch_us)).type(torch.LongTensor), requires_grad=False).to(self.device))
        time_embs=self.timedlt_embeddings(Variable(torch.from_numpy(np.array(batch_timedlt)).type(torch.LongTensor), requires_grad=False).to(self.device))
        batchsize = E_embs.shape[0]
        intra_len=E_embs.shape[2]
        
        paddings_mask1=self.make_std_mask(seq.view(-1, intra_len))
        atted_Es,intra_weights = self.intra_attention(E_embs.view(-1, intra_len, self.args.d_model),paddings_mask1)#[batchsize*sessions,items,d_model]
        atted_E = torch.sum(atted_Es, dim=1)
        E_hat = atted_E.view(batchsize, self.args.inter_len,self.args.d_model) #[batchsize, sessions,d_model]

        u_short = E_hat[:,-1,:]

        paddings_mask2 =self.make_std_mask(seq[:,:,0])
        S0 = self.pos_embedding(E_hat)
        for i in range(self.args.blocks):
            S0,inter_weights=self.sublayers(S0,paddings_mask2)
        S1 = self.Norm(S0) #[batchsize, sessions,d_model]

        u_long = torch.sum(S1,dim=1)+U_embs  # [batchsize,d_model]
        long = torch.mm(u_long, self.gate_Wl)
        short = torch.mm(u_short, self.gate_Ws)
        time=torch.mm(time_embs,self.gate_Wt)
        T =torch.sigmoid(long+short+time+self.gate_bias)
        U = T * u_short + (1 - T) * u_long  #[batchsize,d_model]

        if batch_to_predict is None:
            item_embs=self.item_embeddings.weight.data
            score = torch.mm(U, item_embs.t())
            _,x=torch.topk(score, k=50, dim=1)

        else:
            item_embs = self.item_embeddings(Variable(batch_to_predict, requires_grad=False))
            x = torch.squeeze(torch.einsum('bij,bjk->bik', item_embs,torch.unsqueeze(U, 2)))  # [batch_size,items,d_model]*[batch_size,d_model,1]

        return x

    def mask(self,maxlen):
        attn_shape = (1, maxlen, maxlen)
        subsequent_mask=np.ones(attn_shape).astype('uint8')
        #subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')#上三角
        return torch.from_numpy(subsequent_mask) == 1

    def make_std_mask(self,src_seq):
        "Create a mask to hide padding and future "
         #x_mask *= (1 - (x == 0))
        seq_mask = (src_seq != self.padding).unsqueeze(-2)#[batch,1,items] 
        seq_mask = seq_mask & Variable(self.mask(src_seq.size(-1)).type_as(seq_mask.data))
        return seq_mask

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.normal_(-y, y)
                m.bias.data.zero_()













