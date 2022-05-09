from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx, from_networkx


def get_data(path_2_root:str, dataset_name:str, splits:str, split:int) -> tuple:
    authors_edges_papers_general = pd.read_csv(path_2_root + "general_data/SSORC_CS_2010_2021_authors_edges_papers_indices.csv", index_col = 0, converters={"papers_indices": lambda x: x.strip("[]").replace("'","").split(", ")})
    authors_edges_general = pd.read_csv(path_2_root + "general_data/SSORC_CS_2010_2021_authors_edge_list.csv", index_col = 0)
    papers_features_general = pd.read_csv(path_2_root + "general_data/SSORC_CS_2010_2021_papers_features_vectorized_compressed_32.csv", index_col = 0)
    authors_features_general = pd.read_csv(path_2_root + "general_data/SSORC_CS_2010_2021_authors_features.csv", index_col = 0)
    aev = authors_edges_general.values
    edge_to_index = {(aev[i][0], aev[i][1]):i for i in tqdm(range(len(aev)))}
    
    authors_edges_papers = pd.read_csv(path_2_root + "datasets/" + dataset_name + "/" + dataset_name + "_" + "authors_edges_papers_indices.csv", index_col = 0,\
                                   converters={"papers_indices": lambda x: x.strip("[]").replace("'","").split(", ")})
    authors_graph = nx.read_edgelist(path_2_root + "datasets/" + dataset_name + "/" + dataset_name + "_" + "authors.edgelist", create_using = nx.DiGraph)
    citation_graph = nx.read_edgelist(path_2_root + "datasets/" + dataset_name + "/" + dataset_name + "_" + "papers.edgelist", create_using = nx.DiGraph)
    papers_targets = pd.read_csv(path_2_root + "datasets/" + dataset_name + "/" + dataset_name + "_papers_targets.csv", index_col = 0)
    
    path = path_2_root + "datasets/" + dataset_name + "/split_" + str(splits) + "/"
    train_data_a = torch.load(path + dataset_name + '_train_sample_' + str(split) + '.data')
    val_data_a = torch.load(path + dataset_name + '_val_sample_' + str(split) + '.data')
    test_data_a = torch.load(path + dataset_name + '_test_sample_' + str(split) + '.data')
    
    papers_nodes = list(citation_graph.nodes)
    papers_nodes = [int(papers_nodes[i]) for i in range(len(papers_nodes))]
    papers_node_features = papers_features_general.iloc[papers_nodes, :]
    for node in tqdm(citation_graph.nodes):
        citation_graph.nodes[node]['x'] = list(papers_node_features.loc[[int(node)]].values[0])
    authors_nodes = list(authors_graph.nodes)
    authors_nodes = [int(authors_nodes[i]) for i in range(len(authors_nodes))]
    authors_node_features = authors_features_general.loc[authors_nodes]
    for node in tqdm(authors_graph.nodes):
        authors_graph.nodes[node]['x'] = list(authors_node_features.loc[[int(node)]].values[0])
    data_author = from_networkx(authors_graph)
    data_citation = from_networkx(citation_graph)

    train_data_a.x, val_data_a.x, test_data_a.x = data_author.x.float(), data_author.x.float(), data_author.x.float()
    data_citation.x = data_citation.x.float()

    original_a_nodes = list(authors_graph.nodes)
    pyg_id_2_original_id = {i:int(original_a_nodes[i]) for i in range(len(original_a_nodes))}

    original_a_nodes = list(authors_graph.nodes)
    pyg_id_2_original_id = {i:int(original_a_nodes[i]) for i in range(len(original_a_nodes))}

    sAe_t = train_data_a.edge_index.cpu().numpy().T
    sAe_t = [(pyg_id_2_original_id[int(sAe_t[i][0])], pyg_id_2_original_id[int(sAe_t[i][1])]) for i in range(len(sAe_t))]

    authors_edges_papers_sub_2t = [authors_edges_papers["papers_indices"][edge_to_index[sAe_t[i]]] for i in tqdm(range(len(sAe_t)))]
    authors_edges_papers_sub_flat_2t = [str(item) for subarray in authors_edges_papers_sub_2t for item in subarray]
    unique_papers_2t = list(set(authors_edges_papers_sub_flat_2t))
    
    citation_graph_sub = citation_graph.subgraph(unique_papers_2t)
    citation_graph_sub_nodes = list(citation_graph_sub.nodes())
    global_to_local_id_citation = {citation_graph_sub_nodes[i]:i for i in range(len(citation_graph_sub_nodes))}
    authors_graph_sub_nodes = list(authors_graph.nodes())
    global_to_local_id_authors = {authors_graph_sub_nodes[i]:i for i in range(len(authors_graph_sub_nodes))}

    authors_to_papers = dict()
    for i in tqdm(range(len(sAe_t))):
        papers = authors_edges_papers_sub_2t[i]
        author_1, author_2 = sAe_t[i]
        for author in sAe_t[i]:
            if author in authors_to_papers:
                for paper in papers:
                    authors_to_papers[global_to_local_id_authors[str(author)]].add(global_to_local_id_citation[paper])
            else:
                authors_to_papers[global_to_local_id_authors[str(author)]] = set()
                for paper in papers:
                    authors_to_papers[global_to_local_id_authors[str(author)]].add(global_to_local_id_citation[paper])
    
    for node in tqdm(citation_graph_sub.nodes):
        citation_graph_sub.nodes[node]['x'] = list(papers_features_general.loc[[int(node)]].values[0])
    data_citation = from_networkx(citation_graph_sub)
    data_citation.x = data_citation.x.float()

    authors_nodes = list(authors_graph.nodes)
    authors_nodes = [int(authors_nodes[i]) for i in range(len(authors_nodes))]
    authors_node_features = authors_features_general.loc[authors_nodes]
    
    data_author = from_networkx(authors_graph)
    edges_ordered = [(int(data_author.edge_index.T[i][0]), int(data_author.edge_index.T[i][1])) for i in range(len(data_author.edge_index.T))]
    index_to_edge = {i:edges_ordered[i] for i in range(len(edges_ordered))}
    authors_edges_papers_sample = authors_edges_papers_sub_2t
    citation_nodes = list(citation_graph_sub.nodes)
    ownership_dict = {}
    inds_dict = {}
    for i in tqdm(range(len(authors_edges_papers_sample))):
        arr = authors_edges_papers_sample[i]
        collab_embeddings = []
        for j in range(len(arr)):
            ind = citation_nodes.index(arr[j]) # index_outer_2_index_inner[int(arr[j])]
            collab_embeddings.append(ind)
        ownership_dict[i] = i
        inds_dict[i] = collab_embeddings

    embs_dict = inds_dict
    lens = set([len(embs_dict[i]) for i in range(len(embs_dict))])
    batch_dict_x = {}
    batch_dict_owner = {}
    batch_dict_ind = {}
    for i in tqdm(range(len(embs_dict))):
        if (len(embs_dict[i])) in batch_dict_x:
            batch_dict_x[len(embs_dict[i])].append(embs_dict[i])
            batch_dict_owner[len(embs_dict[i])].append(ownership_dict[i])
            batch_dict_ind[len(embs_dict[i])].append(i)
        else:
            batch_dict_x[len(embs_dict[i])], batch_dict_owner[len(embs_dict[i])], batch_dict_ind[len(embs_dict[i])] = [], [], []
            batch_dict_x[len(embs_dict[i])].append(embs_dict[i])
            batch_dict_owner[len(embs_dict[i])].append(ownership_dict[i])
            batch_dict_ind[len(embs_dict[i])].append(i)

    for length in batch_dict_owner:
        batch_dict_owner[length] = [index_to_edge[batch_dict_owner[length][i]] for i in range(len(batch_dict_owner[length]))]

    batch_list_x = list(batch_dict_x.values())
    batch_list_owner = list(batch_dict_owner.values())
    batch_list_ind = list(batch_dict_ind.values())

    papers_targets = papers_targets.values
    aux_targets = []
    for i in tqdm(range(len(batch_list_x))):
        batch = batch_list_x[i]
        values = []
        for j in range(len(batch)):
            values = [papers_targets[batch[j][k]] for k in range(len(batch[j]))]
            values = np.array(values).T
            targets = [max(values[0]), sum(values[1])/len(values[1]), 
                       sum(values[2])/len(values[2]), sum(values[3])/len(values[3]),
                       len(values[0])]
            aux_targets.append(targets)

    batch_list_owner_flat = [edge for batch in batch_list_owner for edge in batch]
    aux_target_dict = {batch_list_owner_flat[i]:aux_targets[i] for i in tqdm(range(len(aux_targets)))}

    train_edges_aux_t, val_edges_aux_t, test_edges_aux_t = train_data_a.edge_label_index.cpu().numpy().T,\
                                                           val_data_a.edge_label_index.cpu().numpy().T,\
                                                           test_data_a.edge_label_index.cpu().numpy().T

    train_edges_aux_t, val_edges_aux_t, test_edges_aux_t = [(train_edges_aux_t[i][0], train_edges_aux_t[i][1]) for i in range(len(train_edges_aux_t))],\
                                                           [(val_edges_aux_t[i][0], val_edges_aux_t[i][1]) for i in range(len(val_edges_aux_t))],\
                                                           [(test_edges_aux_t[i][0], test_edges_aux_t[i][1]) for i in range(len(test_edges_aux_t))]

    def get_aux_targets(train_edges_aux_t: list) -> list:
        aux_train_target = []
        for k in range(len(train_edges_aux_t)):
            if train_edges_aux_t[k] in aux_target_dict:
                aux_train_target.append(aux_target_dict[train_edges_aux_t[k]])
            else:
                aux_train_target.append([0, 0, 0, 0, 0])
        return aux_train_target

    def task_split(aux_train_targets):
        y_q, y_sjr, y_h, y_if, y_n = np.array(aux_train_targets).T
        return torch.Tensor(y_q.T).float().cuda(),\
               torch.Tensor(y_sjr.T).float().cuda(),\
               torch.Tensor(y_h.T).float().cuda(),\
               torch.Tensor(y_if.T).float().cuda(),\
               torch.Tensor(y_n.T).float().cuda()

    aux_train_targets, aux_val_targets, aux_test_targets = get_aux_targets(train_edges_aux_t),\
                                                           get_aux_targets(val_edges_aux_t),\
                                                           get_aux_targets(test_edges_aux_t)

    train_aux_y, test_aux_y = task_split(aux_train_targets), task_split(aux_test_targets)

    train_data_a.aux = train_aux_y
    test_data_a.aux = test_aux_y
    
    return data_citation, train_data_a, val_data_a, test_data_a, authors_to_papers, batch_list_x, batch_list_owner