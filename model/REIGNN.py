import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList, Embedding
from torch.nn import MultiheadAttention
from torch.autograd import Variable

import torch_geometric.transforms as T
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.data import Data

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

class ResLinearBlock(nn.Module):
    def __init__(self, size, link_size):
        super(ResLinearBlock, self).__init__()
        self.linear_1 = nn.Linear(size, size)
        self.linear_2 = nn.Linear(size, size)
        self.linear_3 = nn.Linear(size, link_size)
        self.batch_norm_1 = nn.BatchNorm1d(size)
        self.batch_norm_2 = nn.BatchNorm1d(size)

    def forward(self, x):
        result = self.linear_1(x)
        result = F.relu(self.batch_norm_1(result))
        result = self.linear_2(result)
        result = F.relu(self.batch_norm_2(result))
        result = self.linear_3(x + result)
        return result


class REIGNN(nn.Module):
    def __init__(self, data_c, heads, device,
                 train_data_a, val_data_a, test_data_a,
                 authors_to_papers,
                 cit_layers = 3, latent_size_cit = 128,
                 auth_layers = 3, latent_size_auth = 128,
                 link_size = 128,
                 lstm_num_layers = 1, lstm_hidden_size = 128):
        super(REIGNN, self).__init__()
        
        self.data_c  = data_c
        self.train_data_a, self.val_data_a, self.test_data_a = train_data_a, val_data_a, test_data_a
        self.authors_to_papers = authors_to_papers
        self.device = device
        self.latent_size_cit = latent_size_cit
        
        # convolutions on citation graph
        self.convs_c = ModuleList()
        conv_с = GATv2Conv(data_c.x.shape[1], latent_size_cit, heads = heads)
        self.convs_c.append(conv_с)
        
        for _ in range(cit_layers):
            conv_с = GATv2Conv(latent_size_cit, latent_size_cit, heads = heads)
            self.convs_c.append(conv_с)
    
        # aggregation
        self.input_size = latent_size_cit+19
        self.num_layers = lstm_num_layers
        self.hidden_size = latent_size_auth
        
        self.lstm = nn.LSTM(input_size=latent_size_auth, hidden_size=latent_size_auth,
                    num_layers=lstm_num_layers, batch_first=True)
        
        self.aggregator = nn.Linear(self.input_size, self.input_size)
        
        # convolutions on co-authorship graph  
        self.convs_a = ModuleList()
        self.batch_norms = ModuleList()
        
        self.pre_conv = nn.Linear(self.input_size, latent_size_auth)
        
        for _ in range(auth_layers):
            conv_a = GATv2Conv(latent_size_auth, latent_size_auth, heads = heads)
            self.convs_a.append(conv_a)
            self.batch_norms.append(BatchNorm(latent_size_auth))
        
        # post link prediction layers
        self.post_lp_layers = ModuleList()
        hidden_post_lp = nn.Linear(latent_size_auth, 1)
        self.post_lp_layers.append(hidden_post_lp)
        
        for _ in range(4):
            hidden_post_lp = nn.Linear(link_size, 1)
            self.post_lp_layers.append(hidden_post_lp)
        
        # multitask
        self.hidden_q1 = ResLinearBlock(latent_size_auth, link_size)
        self.hidden_q2 = nn.Linear(link_size, link_size)
        
        self.hidden_if1 = ResLinearBlock(latent_size_auth, link_size)
        self.hidden_if2 = nn.Linear(link_size, link_size)
        
        self.hidden_hi1 = ResLinearBlock(latent_size_auth, link_size)
        self.hidden_hi2 = nn.Linear(link_size, link_size)

        self.hidden_sjr1 = ResLinearBlock(latent_size_auth, link_size)
        self.hidden_sjr2 = nn.Linear(link_size, link_size)
        
        self.hidden_number1 = ResLinearBlock(latent_size_auth, link_size)
        self.hidden_number2 = nn.Linear(link_size, link_size)


    def forward(self, sample, batch_list_x, batch_list_owner, operator = "hadamard"):
        def cp(z, edge_index):
            return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        def l1(z, edge_index):
            return (torch.abs(z[edge_index[0]] - z[edge_index[1]]))

        def l2(z, edge_index):
            return (torch.pow(z[edge_index[0]] - z[edge_index[1]], 2))

        def hadamard(z, edge_index):
            return (z[edge_index[0]] * z[edge_index[1]])
       
        def summ(z, edge_index):
            return (z[edge_index[0]] + z[edge_index[1]])
        
        x_a = self.train_data_a.x
        if True:
            x_c = self.data_c.x
            for conv in self.convs_c:
                x_c = F.relu(conv(x_c, self.data_c.edge_index))

            counter = 0
            x = torch.zeros(x_a.shape[0], x_a.shape[1] + x_c.shape[1]).to(self.device)
            for i in range(len(x_a)):
                if i in self.authors_to_papers:
                    collab_emb = sum(x_c[list(self.authors_to_papers[i])])
                else:
                    counter += 1
                    collab_emb = torch.zeros(self.latent_size_cit).to(self.device)
                x[i] = torch.cat((x_a[i].unsqueeze(0), collab_emb.unsqueeze(0)), 1)

        convolutions = []

        x = self.pre_conv(x)
        convolutions.append(x)
        for conv, batch_norm in zip(self.convs_a, self.batch_norms):
            x = F.relu(conv(x, self.train_data_a.edge_index))
            convolutions.append(x)
        horizontal = []
        for j in range(len(convolutions[0])):
            horizontal.append([convolutions[i][j] for i in range(len(convolutions))])
            horizontal[j] = torch.stack(horizontal[j])
        emb_seqs_t = torch.stack(horizontal)
        h_0 = Variable(torch.zeros(
        self.num_layers, emb_seqs_t.size(0), self.hidden_size)).to(self.device)

        c_0 = Variable(torch.zeros(
        self.num_layers, emb_seqs_t.size(0), self.hidden_size)).to(self.device)

        ula, (h_out, _) = self.lstm(emb_seqs_t, (h_0, c_0))        
        
        h_out = h_out.view(-1, self.hidden_size)

        q = self.hidden_q1(x)
        q = self.hidden_q2(q)
        
        ifact = self.hidden_if1(x)
        ifact = self.hidden_if2(ifact)
        
        hi = self.hidden_hi1(x)
        hi = self.hidden_hi2(hi)
        
        sjr = self.hidden_sjr1(x)
        sjr = self.hidden_sjr2(sjr)
        
        number = self.hidden_number1(x)
        number = self.hidden_number2(number)
        
        operator_dict = {"cp": cp, "l1": l1, "l2": l2, "hadamard": hadamard, "summ": summ}
        embedding_operator = operator_dict[operator]
        
        edge_index = sample.edge_label_index
        
        link_embeddings, sjr_embeddings, hi_embeddings, if_embeddings, number_embeddings = embedding_operator(h_out, edge_index),\
                                                                        embedding_operator(sjr, edge_index),\
                                                                        embedding_operator(hi, edge_index),\
                                                                        embedding_operator(ifact, edge_index),\
                                                                        embedding_operator(number, edge_index)
        
        embeddings = [link_embeddings, sjr_embeddings, hi_embeddings, if_embeddings, number_embeddings]
        if embedding_operator != cp:
            for i in range(len(embeddings)):
                embeddings[i] = self.post_lp_layers[i](embeddings[i]).squeeze(-1)
            
        return embeddings