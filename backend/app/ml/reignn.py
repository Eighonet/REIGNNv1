from typing import List

import torch
import torch_geometric
from torch import nn
from torch.autograd import Variable
from torch.nn import ModuleList
import torch.nn.functional as F

# noinspection PyProtectedMember
from torch_geometric.nn import GATv2Conv, BatchNorm


class gs_sum_concatenation_gs(nn.Module):
    def __init__(
        self,
        data_c,
        heads,
        train_data_a,
        val_data_a,
        test_data_a,
        cit_layers=3,
        latent_size_cit: int = 128,
        auth_layers=3,
        latent_size_auth=128,
        link_size=128,
        lstm_num_layers=1,
    ):
        super(gs_sum_concatenation_gs, self).__init__()

        self.data_c = data_c
        self.device = torch.device("cpu")
        self.train_data_a, self.val_data_a, self.test_data_a = (
            train_data_a,
            val_data_a,
            test_data_a,
        )

        # convolutions on citation graph
        self.convs_c = ModuleList()
        conv_c = GATv2Conv(data_c.x.shape[1], latent_size_cit, heads=heads)
        self.convs_c.append(conv_c)

        for _ in range(cit_layers):
            conv_c = GATv2Conv(latent_size_cit, latent_size_cit, heads=heads)
            self.convs_c.append(conv_c)

        # aggregation
        self.latent_size_cit: int = latent_size_cit
        self.input_size: int = latent_size_cit + 19
        self.num_layers: int = lstm_num_layers
        self.hidden_size: int = latent_size_auth

        self.lstm = nn.LSTM(
            input_size=latent_size_auth,
            hidden_size=latent_size_auth,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.aggregator = nn.Linear(self.input_size, self.input_size)

        # convolutions on co-authorship graph
        self.convs_a = ModuleList()
        self.batch_norms = ModuleList()

        self.pre_conv = nn.Linear(self.input_size, latent_size_auth)

        for _ in range(auth_layers):
            conv_a = GATv2Conv(latent_size_auth, latent_size_auth, heads=heads)
            self.convs_a.append(conv_a)
            self.batch_norms.append(BatchNorm(latent_size_auth))

        # post link prediction layers
        self.post_lp_layers = ModuleList()
        hidden_post_lp = nn.Linear(latent_size_auth, 1)
        self.post_lp_layers.append(hidden_post_lp)

        for _ in range(5):
            hidden_post_lp = nn.Linear(link_size, 1)
            self.post_lp_layers.append(hidden_post_lp)

        # multitask
        self.hidden_q1 = nn.Linear(latent_size_auth, link_size)
        self.hidden_q2 = nn.Linear(link_size, link_size)

        self.hidden_if1 = nn.Linear(latent_size_auth, link_size)
        self.hidden_if2 = nn.Linear(link_size, link_size)

        self.hidden_hi1 = nn.Linear(latent_size_auth, link_size)
        self.hidden_hi2 = nn.Linear(link_size, link_size)

        self.hidden_sjr1 = nn.Linear(latent_size_auth, link_size)
        self.hidden_sjr2 = nn.Linear(link_size, link_size)

        self.hidden_number1 = nn.Linear(latent_size_auth, link_size)
        self.hidden_number2 = nn.Linear(link_size, link_size)

    def forward(
        self,
        sample: torch_geometric.data.Data,
        sample_a: torch_geometric.data.Data,
        sample_c: torch_geometric.data.Data,
        authors_to_papers,
        operator="hadamard",
    ) -> List[torch.Tensor]:
        self.device = torch.device("cpu")
        self.latent_size_cit = self.input_size - 19

        def cp(z, edge_index_):
            return (z[edge_index_[0]] * z[edge_index_[1]]).sum(dim=-1)

        def l1(z, edge_index_):
            return torch.abs(z[edge_index_[0]] - z[edge_index_[1]])

        def l2(z, edge_index_):
            return torch.pow(z[edge_index_[0]] - z[edge_index_[1]], 2)

        def hadamard(z, edge_index_):
            return z[edge_index_[0]] * z[edge_index_[1]]

        def summ(z, edge_index_):
            return z[edge_index_[0]] + z[edge_index_[1]]

        x_a = sample_a.x
        x_c = sample_c.x
        for conv in self.convs_c:
            x_c = F.relu(conv(x_c, sample_c.edge_index))

        counter = 0
        x = torch.zeros(x_a.shape[0], x_a.shape[1] + x_c.shape[1]).to(self.device)
        for i in range(len(x_a)):
            if i in authors_to_papers:
                collab_emb = sum(x_c[list(authors_to_papers[i])])
                if len(authors_to_papers[i]) == 0:
                    collab_emb = torch.zeros(self.latent_size_cit).to(self.device)
            else:
                counter += 1
                collab_emb = torch.zeros(self.latent_size_cit).to(self.device)
            x[i] = F.relu(
                self.aggregator(
                    torch.cat((x_a[i].unsqueeze(0), collab_emb.unsqueeze(0)), 1)
                )
            )

        convolutions = []

        x = self.pre_conv(x)
        convolutions.append(x)
        for conv, batch_norm in zip(self.convs_a, self.batch_norms):
            x = F.relu(conv(x, sample_a.edge_index))
            convolutions.append(x)
        horizontal = []
        for j in range(len(convolutions[0])):
            horizontal.append([convolutions[i][j] for i in range(len(convolutions))])
            horizontal[j] = torch.stack(horizontal[j])
        emb_seqs_t = torch.stack(horizontal)

        h_0 = Variable(
            torch.zeros(self.num_layers, emb_seqs_t.size(0), self.hidden_size)
        ).to(self.device)

        c_0 = Variable(
            torch.zeros(self.num_layers, emb_seqs_t.size(0), self.hidden_size)
        ).to(self.device)

        ula, (h_out, _) = self.lstm(emb_seqs_t, (h_0, c_0))

        x = h_out.view(-1, self.hidden_size)
        # q = self.hidden_q1(x)
        # q = self.hidden_q2(q)

        ifact = self.hidden_if1(x)
        ifact = self.hidden_if2(ifact)

        hi = self.hidden_hi1(x)
        hi = self.hidden_hi2(hi)

        sjr = self.hidden_sjr1(x)
        sjr = self.hidden_sjr2(sjr)

        number = self.hidden_number1(x)
        number = self.hidden_number2(number)

        operator_dict = {
            "cp": cp,
            "l1": l1,
            "l2": l2,
            "hadamard": hadamard,
            "summ": summ,
        }
        embedding_operator = operator_dict[operator]

        edge_index = sample.edge_index

        (
            link_embeddings,
            sjr_embeddings,
            hi_embeddings,
            if_embeddings,
            number_embeddings,
        ) = (
            embedding_operator(x, edge_index),
            embedding_operator(sjr, edge_index),
            embedding_operator(hi, edge_index),
            embedding_operator(ifact, edge_index),
            embedding_operator(number, edge_index),
        )

        embeddings = [
            link_embeddings,
            sjr_embeddings,
            hi_embeddings,
            if_embeddings,
            number_embeddings,
        ]
        if embedding_operator != cp:
            for i in range(len(embeddings)):
                embeddings[i] = self.post_lp_layers[i](embeddings[i]).squeeze(-1)

        return embeddings
