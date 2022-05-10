import os
from os import listdir
from tqdm import tqdm
import json
from collections import Counter
import itertools
import random

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_hub as hub
import networkx as nx


def get_papers_dict(areas:list, year_start:int, year_end:int, path_2_unpacked:str = './unpacked') -> dict:
    """
    Return dictionary of papers in form id: [properties]. 
    
    """
    papers_dict = {}
    files = [f for f in listdir(path_2_unpacked)]
    areas = set(areas)
    
    for j in tqdm(range(len(files))):
        with open(path_2_unpacked + '/' + files[j]) as f:
            lines = f.readlines()
        cs_papers_local = []
        for i in (range(len(lines))):
            paper = json.loads(lines[i])
            if not paper["year"]:
                continue
            if len(set(paper["fieldsOfStudy"]).intersection(areas)) > 0 \
            and paper["year"] >= year_start \
            and paper["year"] <= year_end \
            and len(paper["inCitations"]) > 0 \
            and len(paper["outCitations"]) > 0 \
            and len(paper["doi"]) > 0 \
            and len(paper["paperAbstract"]) > 0 \
            and len(paper["title"]) > 0 \
            and len(paper["journalName"]) > 0:
                papers_dict[paper["id"]] = paper
                cs_papers_local.append(paper)
    return papers_dict

def get_edge_list(papers_dict:dict) -> list:
    edge_list = []
    for paper_id in tqdm(papers_dict):
        paper = papers_dict[paper_id]
        paper_cit = paper['outCitations']
        for j in range(len(paper_cit)):
            if (paper_cit[j] in papers_dict):
                edge_list.append([paper_id, paper_cit[j]])
    return edge_list
                
def get_data(papers_dict:dict, edge_list:list, dataset_name:str) -> list:
    no_id_counter = 0
    edge_dict = {} # keys -- edge list (author_1_id, author_2_id), values -- corresponding papers ids
    authors_dict = {} # keys -- author_id, values -- papers ids
    authors_interests = {} # keys -- author_id, values -- papers ids
    for paper_id in tqdm(papers_dict):
        paper = papers_dict[paper_id]
        itertools.permutations(paper["authors"], 2)
        ids = []
        for author in paper["authors"]:
            if len(author['ids']) == 1:
                author_id = author['ids'][0]
                ids.append(author_id)
                areas = paper['fieldsOfStudy']
                if author_id in authors_dict:
                    authors_dict[author_id][1].add(paper_id)
                    for area in areas:
                        authors_interests[author_id][1].add(area)
                else:
                    authors_dict[author_id] = [author['name'], {paper_id}]
                    authors_interests[author_id] = [author_id, set()]
                    for area in areas:
                        authors_interests[author_id][1].add(area)
            else:
                no_id_counter += 1
        authors_pairs = list(itertools.combinations(ids, 2))
        for i in range(len(authors_pairs)):
            if authors_pairs[i] in edge_dict:
                edge_dict[authors_pairs[i]].append(paper_id)
            else:
                edge_dict[authors_pairs[i]] = [paper_id]
    
    authors_interests_list = list(authors_interests.values())
    df = pd.DataFrame(np.array(authors_interests_list), columns = ["author_id", "interests"])
    
    try:
        os.mkdir("processed_data")
    except:
        pass
    
    papers_df = pd.DataFrame(list(papers_dict.values()))
    papers_features = papers_df.drop(["inCitations", "outCitations"], axis = 1)
    papers_features.to_csv("processed_data/"  + dataset_name + "_papers_features.csv")
    
    authors_features = df.drop('interests', 1).join(df.interests.str.join('|').str.get_dummies())
    authors_features.to_csv("processed_data/" + dataset_name + "_authors_features.csv")
    
    edge_dict_values = list(edge_dict.values())
    authors_papers = pd.DataFrame(np.array(edge_dict_values), columns = ["papers_ids"])
    authors_papers.to_csv("processed_data/" + dataset_name + "_authors_edges_papers.csv")
    
    edge_dict_keys = list(edge_dict.keys())
    authors_edges = pd.DataFrame(edge_dict_keys, columns = ["from", "to"])
    authors_edges.to_csv("processed_data/" + dataset_name + "_authors_edge_list.csv")
    
    papers_edges = pd.DataFrame(edge_list, columns = ["from", "to"])
    papers_edges.to_csv("processed_data/" + dataset_name + "_papers_edge_list.csv")

    return [papers_features, authors_features, authors_papers, authors_edges, papers_edges]

def parse_global_dataset(areas, year_start, year_end, dataset_name:str = "test_dataset") -> list:
    papers_dict = get_papers_dict(areas,  year_start, year_end)
    edge_list = get_edge_list(papers_dict)
    global_dataset = get_data(papers_dict, edge_list, dataset_name)
    return global_dataset

def preprocessing(global_dataset:list, dataset_name:str = "test_dataset") -> list:
    papers_features, authors_features, authors_papers, authors_edges, papers_edges = global_dataset
    authors = []
    
    papers_id = papers_features["id"]
    id_to_index_id = {papers_id[i]: i for i in tqdm(range(len(papers_id)))}

    authors_papers_unzipped = authors_papers["papers_ids"]

    authors_papers_indexed = [
        [
            id_to_index_id[authors_papers_unzipped[i][j]]
            for j in range(len(authors_papers_unzipped[i]))
        ]
        for i in tqdm(range(len(authors_papers_unzipped)))
    ]

    authors_papers_indexed_str = [
        str(authors_papers_indexed[i]) for i in tqdm(range(len(authors_papers_indexed)))
    ]
    
    authors_edges_papers_indices = pd.DataFrame(authors_papers_indexed_str, columns=["papers_indices"])
    authors_edges_papers_indices.to_csv(
        "processed_data/" + dataset_name + "_authors_edges_papers_indices.csv"
    )

    df = papers_features[
        papers_features[
            ["id", "title", "paperAbstract", "year", "journalName", "fieldsOfStudy"]
        ].notna()
    ]

    papers_features_abstracts = list(papers_features["paperAbstract"])
    papers_features_abstracts = [
        str(papers_features_abstracts[i]) for i in range(len(papers_features_abstracts))
    ]

    papers_features["paperAbstract"] = papers_features["paperAbstract"].fillna(
        "No abstract provided"
    )
    
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    vectorized_abstracts = []
    for i in tqdm(range(len(papers_features_abstracts))):
        abstract = papers_features_abstracts[i]
        vectorized_abstracts.append(model([abstract])[0])

    vectorized_abstracts_list = [
        vectorized_abstracts[i].numpy() for i in tqdm(range(len(vectorized_abstracts)))
    ]

    vectorized_abstracts_df = pd.DataFrame(vectorized_abstracts_list)

    print('PCA started its work.')
    pca = PCA(n_components=32)
    pca_result = pca.fit_transform(vectorized_abstracts_df)
    print('PCA ended its work.')
    
    compressed_paper_features = pd.DataFrame(pca_result)
    compressed_paper_features.to_csv(
        "processed_data/" + dataset_name + "_papers_features_vectorized_compressed_32.csv"
    )

    papers_edge_list_indexed = papers_edges.values
    for i in tqdm(range(len(papers_edge_list_indexed))):
        pair = papers_edge_list_indexed[i]
        for j in range(len(pair)):
            pair[j] = id_to_index_id[pair[j]]

    papers_edge_list_indexed_np = pd.DataFrame(papers_edge_list_indexed)

    papers_edge_list_indexed_np.to_csv(
        "processed_data/" + dataset_name + "_papers_edge_list_indexed.csv"
    )
    
    return [authors_edges_papers_indices, compressed_paper_features, papers_edge_list_indexed_np]

def extract_subgraph(global_dataset:list, processed_data:list, subgraph_name:str, nodes_number:int = 1000):
    
    def get_nx_graph(edge_list):
        aev = edge_list.values
        edge_to_index = {(aev[i][0], aev[i][1]): i for i in tqdm(range(len(aev)))}
        edges_list_t = list(edge_to_index.keys())
        return edge_to_index, nx.DiGraph((x, y) for (x, y) in tqdm(Counter(edges_list_t)))


    def get_subraph(N, source: int, depth_limit: int = 4):
        nodes = list(nx.dfs_preorder_nodes(N, source=source, depth_limit=depth_limit))
        H = N.subgraph(nodes)
        return H
    
    authors_edges_papers, compressed_paper_features, papers_edge_list_indexed_np = processed_data
    papers_features, authors_features, authors_papers, authors_edges, papers_edges = global_dataset
    
    edge_to_index_A, A = get_nx_graph(authors_edges)
    edge_to_index_G, G = get_nx_graph(papers_edge_list_indexed_np)
    
    try:
        authors_edges_papers['papers_indices'] = authors_edges_papers['papers_indices'].apply(lambda x: x.replace('[', '').replace(']', '').split(','))
    except:
        pass
    
    depth_limit, ready_flag, sub_A = 3, 0, ""
    for i in range(depth_limit, 15):
        if ready_flag == 0:
            for i in range(10):
                source = random.choice(list(A.nodes()))
                sub_A = get_subraph(A, source, depth_limit=i)
                if len(sub_A.nodes) >= nodes_number:
                     ready_flag = 1
        else:
            break

    sub_A_edges = list(sub_A.edges())
    
    authors_edges_papers_sub = [
        authors_edges_papers["papers_indices"][edge_to_index_A[sub_A_edges[i]]]
        for i in tqdm(range(len(sub_A_edges)))
    ]

    authors_edges_papers_sub_flat = [
        int(item) for subarray in authors_edges_papers_sub for item in subarray
    ]
    unique_papers = list(set(authors_edges_papers_sub_flat))
    
    papers_to_delete_initial = list(set(unique_papers) - set(G.nodes))
    G_sub = G.subgraph(unique_papers)
    G_sub_nodes = list(G_sub.nodes())
    
    
    
    papers_out_lcc = papers_to_delete_initial
    collabs_indices_to_delete = []

    for i in tqdm(range(len(papers_out_lcc))):
        for j in range(len(authors_edges_papers_sub)):
            #        if str(1745104) in authors_edges_papers_sub[j]:
            #            jj.append(j)
            if str(papers_out_lcc[i]) in authors_edges_papers_sub[j]:
                del authors_edges_papers_sub[j][
                    authors_edges_papers_sub[j].index(str(papers_out_lcc[i]))
                ]
                if len(authors_edges_papers_sub[j]) == 0:
                    collabs_indices_to_delete.append(j)

    A_sub_clear = nx.DiGraph(sub_A)
    A_sub_clear_edges = list(A_sub_clear.edges())

    for i in tqdm(range(len(collabs_indices_to_delete))):
        edge = A_sub_clear_edges[collabs_indices_to_delete[i]]
        if edge not in A_sub_clear_edges:
            print("error")

        A_sub_clear.remove_edge(*edge)

    authors_edges_papers_sub_clear = [
        authors_edges_papers_sub[i]
        for i in range(len(authors_edges_papers_sub))
        if len(authors_edges_papers_sub[i]) > 0
    ]


    A_sub_clear_edges_check = list(A_sub_clear.edges())

    authors_edges_papers_sub_2 = [
        authors_edges_papers["papers_indices"][edge_to_index_A[A_sub_clear_edges_check[i]]]
        for i in tqdm(range(len(A_sub_clear_edges_check)))
    ]

    authors_edges_papers_sub_2 = [
        authors_edges_papers["papers_indices"][edge_to_index_A[A_sub_clear_edges_check[i]]]
        for i in tqdm(range(len(A_sub_clear_edges_check)))
    ]
    authors_edges_papers_sub_flat_2 = [
        int(item) for subarray in authors_edges_papers_sub_2 for item in subarray
    ]
    unique_papers_2 = list(set(authors_edges_papers_sub_flat_2))

    G_sub_clear = G_sub
    
    try:
        os.mkdir('datasets')
    except:
        pass
    
    try:
        os.mkdir('datasets/' + subgraph_name)
    except:
        pass
    
    nx.write_edgelist(
        G_sub_clear,
        "datasets/" + subgraph_name + "/" + subgraph_name + "_" + "papers.edgelist",
    )

    nx.write_edgelist(
        A_sub_clear,
        "datasets/" + subgraph_name + "/" + subgraph_name + "_" + "authors.edgelist",
    )

    authors_edges_papers.to_csv(
        "datasets/"
        + subgraph_name
        + "/"
        + subgraph_name
        + "_"
        + "authors_edges_papers_indices.csv"
    )