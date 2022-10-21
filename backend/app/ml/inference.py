from pathlib import Path
from typing import List, Any, Tuple
import random

import numpy as np
import torch
import torch_geometric.data.data
from torch_geometric.utils import from_networkx
from loguru import logger
import networkx as nx

from .datafactory import DataFactory


class GraphInference:
    def __init__(self, root_path=Path("data/"), global_dataset="SSORC_CS_2010_2021"):
        self.root_path = root_path
        self.global_dataset = global_dataset

        self.data: DataFactory = DataFactory()
        logger.info("Loading PyTorch Model...")
        self.model = torch.load(
            self.root_path / "REIGNN_test_v4.pt", map_location=torch.device("cpu")
        )
        logger.info("PyTorch model was loaded!")

        self.A_sub: nx.DiGraph = nx.DiGraph()
        self.A_sub_edges: list = list()

    def get_authors(self, authors: List[int] = None) -> np.ndarray:
        if not authors:
            target_authors_ids = [
                random.choice(list(self.data.A.nodes())) for _ in range(20)
            ]
        else:
            target_authors_ids = authors

        self.get_subgraph_from_target_authors(
            target_authors_ids=target_authors_ids, depth_limit=3
        )
        unique_papers = self.get_unique_papers_from_graph_edges()
        self.citation_subgraph_extraction(unique_papers=unique_papers)

        return np.array(target_authors_ids)

    def get_subgraph_from_target_authors(
        self, target_authors_ids=None, depth_limit=4
    ) -> None:
        if target_authors_ids is None:
            target_authors_ids = self.get_authors()
        target_nodes = []
        for i in range(len(target_authors_ids)):
            sub_A_c = self.data.get_subgraph(
                self.data.A, target_authors_ids[i], depth_limit=depth_limit
            )
            target_nodes += list(sub_A_c.nodes())

        self.A_sub = self.data.A.subgraph(target_nodes)
        self.A_sub_edges = list(self.A_sub.edges())

    def get_unique_papers_from_graph_edges(self) -> List[int]:
        # logger.debug(
        #     f'authors_edges_papers : {len(self.data.authors_edges_papers["papers_indices"])} '
        #     f'{(self.data.authors_edges_papers["papers_indices"])[:2]}'
        # )
        authors_edges_papers_sub = [
            self.data.authors_edges_papers["papers_indices"][
                self.data.edge_to_index_A[self.A_sub_edges[i]]
            ]
            for i in range(len(self.A_sub_edges))
        ]
        authors_edges_papers_sub_flat = [
            int(item) for subarray in authors_edges_papers_sub for item in subarray
        ]
        unique_papers = list(set(authors_edges_papers_sub_flat))
        # logger.debug(f"unique_papers : {len(unique_papers)} {unique_papers[:2]}")
        return unique_papers

    def citation_subgraph_extraction(self, unique_papers: List[Any] = None):
        G_sub = self.data.G.subgraph(unique_papers)
        # logger.debug(f"G_sub : {G_sub}")
        G_sub_clear = G_sub

        for node in G_sub_clear.nodes:
            self.data.G.add_edge(node, node)

        A_sub_clear = self.A_sub

        # def get_graph_properties(H):
        #     print(
        #         "Nodes in in the final subgraph: ",
        #         len(H.nodes()),
        #         "\nEdges in the final subgraph: ",
        #         len(H.edges()),
        #     )
        #     print("Diameter: ", nx.diameter(H.to_undirected()))
        #     print(
        #         "Average clustering coefficient: ",
        #         nx.average_clustering(H.to_undirected()),
        #     )
        #     return len(H.nodes())

        # p1 = get_graph_properties(A_sub_clear)
        # p2 = get_graph_properties(G_sub_clear)
        # nodes_author, nodes_citation = p1, p2

        # self.data.dataset_name = (
        #     "SSORC_CS_10_21_"
        #     + str(nodes_author)
        #     + "_"
        #     + str(nodes_citation)
        #     + "_unfiltered"
        # )
        # dataset_path = self.root_path / "datasets" / self.data.dataset_name

        self.data.authors_graph = A_sub_clear
        self.data.citation_graph = G_sub_clear

        # nx.write_edgelist(
        #     G_sub_clear,
        #     dataset_path / (self.data.dataset_name + "_" + "papers.edgelist"),
        # )
        # nx.write_edgelist(
        #     A_sub_clear,
        #     dataset_path / (self.data.dataset_name + "_" + "authors.edgelist"),
        # )

        # self.data.authors_edges_papers.to_csv(
        #     dataset_path
        #     / (self.data.dataset_name + "_" + "authors_edges_papers_indices.csv")
        # )

    def get_train_data(
        self,
    ) -> Tuple[torch_geometric.data.Data, torch_geometric.data.Data, dict]:
        # sAe = list(self.authors_graph.edges)
        # sAe = [(int(sAe[i][0]), int(sAe[i][1])) for i in range(len(sAe))]

        # authors_edges_papers_sub_2 = [self.authors_edges_papers["papers_indices"][self.edge_to_index[sAe[i]]] for i in
        #                               range(len(sAe))]
        # authors_edges_papers_sub_flat_2 = [int(item) for subarray in authors_edges_papers_sub_2 for item in subarray]
        # unique_papers_2 = list(set(authors_edges_papers_sub_flat_2))

        # cgn = list(self.citation_graph.nodes())
        # cgn = [int(cgn[i]) for i in range(len(cgn))]

        papers_nodes = list(self.data.citation_graph.nodes)
        papers_nodes = [int(papers_nodes[i]) for i in range(len(papers_nodes))]
        papers_node_features = self.data.papers_features.iloc[papers_nodes, :]
        for node in self.data.citation_graph.nodes:
            self.data.citation_graph.nodes[node]["x"] = list(
                papers_node_features.loc[[int(node)]].values[0]
            )
        authors_nodes = list(self.data.authors_graph.nodes)
        authors_nodes = [int(authors_nodes[i]) for i in range(len(authors_nodes))]
        authors_node_features = self.data.authors_features.loc[authors_nodes]

        for node in self.data.authors_graph.nodes:
            self.data.authors_graph.nodes[node]["x"] = list(
                authors_node_features.loc[[int(node)]].values[0]
            )

        data_author: torch_geometric.data.Data = from_networkx(self.data.authors_graph)
        train_data_a = torch_geometric.data.Data(
            x=data_author.x.float(), edge_index=data_author.edge_index
        )

        # logger.debug(f"train_data_a : {train_data_a}")

        original_a_nodes = list(self.data.authors_graph.nodes)
        self.data.pyg_id_2_original_id = {
            i: int(original_a_nodes[i]) for i in range(len(original_a_nodes))
        }
        self.data.pyg_original_id_2_id = {
            int(original_a_nodes[i]): i for i in range(len(original_a_nodes))
        }

        sAe_t = train_data_a.edge_index.cpu().numpy().T
        sAe_t = [
            (
                self.data.pyg_id_2_original_id[int(sAe_t[i][0])],
                self.data.pyg_id_2_original_id[int(sAe_t[i][1])],
            )
            for i in range(len(sAe_t))
        ]

        authors_edges_papers_sub_2t = [
            self.data.authors_edges_papers["papers_indices"][
                self.data.edge_to_index_A[sAe_t[i]]
            ]
            for i in range(len(sAe_t))
        ]
        authors_edges_papers_sub_flat_2t = [
            int(item) for subarray in authors_edges_papers_sub_2t for item in subarray
        ]
        unique_papers_2t = list(set(authors_edges_papers_sub_flat_2t))

        # logger.debug(
        #     f"authors_edges_papers_sub_2t : {len(authors_edges_papers_sub_2t)} {authors_edges_papers_sub_2t[:2]}"
        # )
        # logger.debug(
        #     f"authors_edges_papers_sub_flat_2t : "
        #     f"{len(list(authors_edges_papers_sub_flat_2t))} {list(authors_edges_papers_sub_flat_2t)[:2]}"
        # )
        # logger.debug(
        #     f"unique_papers_2t : {len(unique_papers_2t)} {unique_papers_2t[:2]}"
        # )

        citation_graph_sub = self.data.citation_graph.subgraph(unique_papers_2t)
        citation_graph_sub_nodes = list(citation_graph_sub.nodes())
        global_to_local_id_citation = {
            citation_graph_sub_nodes[i]: i for i in range(len(citation_graph_sub_nodes))
        }
        authors_graph_sub_nodes = list(self.data.authors_graph.nodes())
        global_to_local_id_authors = {
            authors_graph_sub_nodes[i]: i for i in range(len(authors_graph_sub_nodes))
        }

        authors_to_papers: dict = dict()
        for i in range(len(sAe_t)):
            papers = authors_edges_papers_sub_2t[i]
            # author_1, author_2 = sAe_t[i]
            for author in sAe_t[i]:
                if author not in authors_to_papers:
                    if str(author) not in global_to_local_id_authors:
                        global_to_local_id_authors[str(author)] = (
                            max(list(global_to_local_id_authors.values())) + 1
                        )
                    authors_to_papers[global_to_local_id_authors[str(author)]] = set()
                for paper in papers:
                    if paper in global_to_local_id_citation:
                        authors_to_papers[global_to_local_id_authors[str(author)]].add(
                            global_to_local_id_citation[paper]
                        )

        # logger.debug(
        #     f"citation_graph_sub : {len(list(citation_graph_sub.nodes()))} {list(citation_graph_sub.nodes())[:2]}"
        # )
        for node in citation_graph_sub.nodes:
            citation_graph_sub.nodes[node]["x"] = list(
                self.data.papers_features.loc[[int(node)]].values[0]
            )
        data_citation: torch_geometric.data.Data = from_networkx(citation_graph_sub)

        try:
            # logger.debug(f"data_citation.x : {data_citation.x}")
            # logger.debug(f"data_citation.edge_index : {data_citation.edge_index}")
            # logger.debug(f"data_citation : {data_citation}")

            data_citation = torch_geometric.data.Data(
                x=data_citation.x.float(), edge_index=data_citation.edge_index
            )
        except Exception as e:
            logger.error(f"error : {e}")
            for node in self.data.citation_graph.nodes:
                self.data.citation_graph.nodes[node]["x"] = list(
                    self.data.papers_features.loc[[int(node)]].values[0]
                )
            data_citation = from_networkx(self.data.citation_graph)

            # logger.debug(f"data_citation : {data_citation}")

            data_citation = torch_geometric.data.Data(
                x=data_citation.x.float(), edge_index=data_citation.edge_index
            )

        return train_data_a, data_citation, authors_to_papers

    def get_predicted_authors(self, predicted_edges: np.ndarray) -> np.ndarray:
        for idx in range(len(predicted_edges)):
            for node in range(2):
                predicted_edges[idx][0][node] = self.data.pyg_id_2_original_id[
                    predicted_edges[idx][0][node]
                ]
        return predicted_edges

    def get_test_data(
        self,
        authors_for_predictions: np.ndarray,
        train_data_a: torch_geometric.data.Data,
    ) -> torch_geometric.data.Data:
        afp = [
            self.data.pyg_original_id_2_id[authors_for_predictions[i]]
            for i in range(len(authors_for_predictions))
            if authors_for_predictions[i] in self.data.pyg_original_id_2_id
        ]

        comb_array_ = np.array(np.meshgrid(afp, afp)).T.reshape(-1, 2)
        comb_array = np.array(
            [
                comb_array_[i]
                for i in range(len(comb_array_))
                if comb_array_[i][0] != comb_array_[i][1]
            ]
        )

        test_data_a = torch_geometric.data.Data(
            x=train_data_a.x, edge_index=torch.Tensor(comb_array.T).long()
        )

        return test_data_a

    def predict(
        self,
        train_data: torch_geometric.data.Data,
        test_data: torch_geometric.data.Data,
        data_citation: torch_geometric.data.Data,
        authors_to_papers: dict,
        probability_threshold=0.75,
    ) -> Tuple[np.ndarray, list, list, list, list, list]:
        z, z_sjr, z_hi, z_ifact, z_numb = self.model(
            test_data,
            train_data,
            data_citation,
            authors_to_papers,
            operator="hadamard",
        )
        logger.debug(f"z : {z}")
        logger.debug(f"z.sigmoid() : {z.sigmoid()}")

        while True:

            predicted_collab_mask: torch.Tensor = z.sigmoid() > probability_threshold
            # logger.debug(f"predicted_collab_mask : {predicted_collab_mask}")

            result_array: np.ndarray = predicted_collab_mask.cpu().detach().numpy()
            # logger.debug(f"result_array : {result_array}")

            result_indices = result_array.nonzero()
            # logger.debug(f"result_indices : {result_indices}")

            if len(result_indices) == 0 or len(result_indices) < len(z) // 2:
                probability_threshold = probability_threshold * 0.95
                if probability_threshold < 0.45:
                    break
                else:
                    continue

            break

        logger.debug(f"final threshold : {probability_threshold}")
        logger.debug(f"final indices length : {len(result_indices)}")
        logger.debug(f"result_indices : {result_indices}")

        result: np.ndarray = np.take(
            test_data.edge_index.cpu().detach().numpy(), result_indices, axis=1
        ).T
        logger.debug(f"result : {result}")

        result_z: list = np.take(
            z.sigmoid().cpu().detach().numpy(), result_indices
        ).tolist()[0]
        logger.debug(f"result_z : {result_z}")

        result_z_sjr: list = np.take(
            z_sjr.cpu().detach().numpy(), result_indices
        ).tolist()[0]
        logger.debug(f"result_z_sjr : {result_z_sjr}")

        result_z_hi: list = np.take(
            z_hi.cpu().detach().numpy(), result_indices
        ).tolist()[0]
        logger.debug(f"result_z_hi : {result_z_hi}")

        result_z_ifact: list = np.take(
            z_ifact.cpu().detach().numpy(), result_indices
        ).tolist()[0]
        logger.debug(f"result_z_ifact : {result_z_ifact}")

        result_z_numb: list = np.take(
            z_numb.cpu().detach().numpy(), result_indices
        ).tolist()[0]
        logger.debug(f"result_z_numb : {result_z_numb}")

        logger.debug(f"result : {result}")

        logger.debug(f"Part one")

        duplicate_indices: List[Tuple[int, int]] = []
        for i in range(result.shape[0] - 1):
            for k in range(i + 1, result.shape[0]):
                # logger.debug(f"{result[i][0][0]} {result[i][0][1]} {result[k][0][0]} {result[k][0][1]}")
                if (
                    result[i][0][0] == result[k][0][1]
                    and result[k][0][0] == result[i][0][1]
                ):
                    duplicate_indices.append((i, k))

        logger.debug(f"{duplicate_indices}")
        logger.debug(f"Part Two")

        for i in duplicate_indices:
            result_z[i[0]] = np.mean([result_z[i[0]], result_z[i[1]]])
            result_z_sjr[i[0]] = np.mean([result_z_sjr[i[0]], result_z_sjr[i[1]]])
            result_z_hi[i[0]] = np.mean([result_z_hi[i[0]], result_z_hi[i[1]]])
            result_z_ifact[i[0]] = np.mean([result_z_ifact[i[0]], result_z_ifact[i[1]]])
            result_z_numb[i[0]] = np.mean([result_z_numb[i[0]], result_z_numb[i[1]]])

        logger.debug(f"Part three")

        logger.debug(f"{len(result)} {len(result_z)}")

        for x in sorted([i[1] for i in duplicate_indices], reverse=True):
            result = np.delete(result, int(x), 0)
            result_z.pop(x)
            result_z_sjr.pop(x)
            result_z_hi.pop(x)
            result_z_ifact.pop(x)
            result_z_numb.pop(x)

        result = self.get_predicted_authors(result)

        logger.debug(f"{len(result)} {len(result_z)}")

        logger.debug(f"result_z : {result_z}")
        logger.debug(f"result_z_sjr : {result_z_sjr}")
        logger.debug(f"result_z_hi : {result_z_hi}")
        logger.debug(f"result_z_ifact : {result_z_ifact}")
        logger.debug(f"result_z_numb : {result_z_numb}")
        logger.debug(f"result : {result}")

        return (
            result,
            result_z,
            result_z_sjr,
            result_z_hi,
            result_z_ifact,
            result_z_numb,
        )
