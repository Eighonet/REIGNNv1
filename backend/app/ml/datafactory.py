from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Set

import networkx as nx
import pandas as pd
import ujson as json
from loguru import logger


class Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DataFactory(metaclass=Singleton):
    def __init__(self, root_path=Path("data/"), global_dataset="SSORC_CS_2010_2021"):
        self.root_path: Path = root_path
        self.global_dataset = global_dataset
        self.dataset_name: str = ""
        self.dataset_path: Path = Path()

        self.A: nx.DiGraph = nx.DiGraph()
        self.G: nx.DiGraph = nx.DiGraph()

        self.authors_graph: nx.DiGraph = nx.DiGraph()
        self.citation_graph: nx.DiGraph = nx.DiGraph()

        self.edge_to_index_A: dict = dict()

        # self.author_2_id: dict = dict()

        # self.edge_to_index: dict = dict()

        self.pyg_id_2_original_id: dict = dict()
        self.pyg_original_id_2_id: dict = dict()

        self.papers_features: pd.DataFrame = pd.DataFrame()
        self.authors_features: pd.DataFrame = pd.DataFrame()

        # self.authors_edges: pd.DataFrame = pd.DataFrame()
        self.authors_edges_papers: pd.DataFrame = pd.DataFrame()

        self.author_data: List[dict] = list()
        self.author_data_ids: Set[int] = set()

        self.initial_graph_loader()

    @staticmethod
    def get_nx_graph(edge_list_path: Path) -> Tuple[dict, nx.DiGraph]:
        edge_list = pd.read_pickle(edge_list_path)
        aev = edge_list.values
        edge_to_index = {(x[0], x[1]): i for i, x in enumerate(aev)}
        digraph = nx.DiGraph()
        digraph.add_edges_from(aev)
        return edge_to_index, digraph

    @staticmethod
    def get_subgraph(N: nx.DiGraph, source: int, depth_limit: int = 3) -> nx.DiGraph:
        nodes = list(nx.dfs_preorder_nodes(N, source=source, depth_limit=depth_limit))
        H = N.subgraph(nodes)
        return H

    def initial_graph_loader(self) -> None:
        start_time = datetime.now()
        self.dataset_path = self.root_path / "datasets" / self.dataset_name

        logger.info("Loading authors_edge_list.pkl...")
        self.edge_to_index_A, self.A = self.get_nx_graph(
            self.root_path / (self.global_dataset + "_authors_edge_list.pkl")
        )
        logger.info("Successfully loaded authors_edge_list.pkl!")

        logger.info("Loading papers_edge_list_indexed.pkl...")
        _, self.G = self.get_nx_graph(
            self.root_path / (self.global_dataset + "_papers_edge_list_indexed.pkl")
        )
        logger.info("Successfully loaded papers_edge_list_indexed.pkl!")

        # logger.info("Loading authors_edge_list.pkl...")
        # self.authors_edges = pd.read_pickle(
        #     self.root_path / (self.global_dataset + "_authors_edge_list.pkl"),
        # )
        # logger.info("Successfully loaded authors_edge_list.pkl!")

        logger.info("Loading authors_edges_papers_indices.pkl...")
        self.authors_edges_papers = pd.read_pickle(
            self.root_path / (self.global_dataset + "_authors_edges_papers_indices.pkl")
        )
        logger.info("Successfully loaded authors_edges_papers_indices.pkl!")

        logger.info("Loading papers_features_vectorized_compressed_32.pkl...")
        self.papers_features = pd.read_pickle(
            self.root_path
            / (self.global_dataset + "_papers_features_vectorized_compressed_32.pkl"),
        )
        logger.info(
            "Successfully loaded papers_features_vectorized_compressed_32.pkl..."
        )

        logger.info("Loading authors_features.pkl...")
        self.authors_features = pd.read_pickle(
            self.root_path / (self.global_dataset + "_authors_features.pkl"),
        )
        logger.info("Successfully loaded authors_features.pkl...")

        # logger.info("Creating edge_to_index...")
        # aev = self.authors_edges.values
        # self.edge_to_index = {(aev[i][0], aev[i][1]): i for i in tqdm(range(len(aev)))}
        # logger.info("Successfully created edge_to_index!")

        # logger.info("Loading author_2_id.pkl...")
        # author_2_id_df = pd.read_pickle(self.root_path / "author_2_id.pkl")
        # authors, ids = author_2_id_df.values[:, 0], author_2_id_df.values[:, 1]
        # self.author_2_id = {authors[i]: ids[i] for i in range(len(authors))}
        # logger.info("Successfully loaded author_2_id.pkl!")

        logger.info("Loading author_data.json...")
        with open(self.root_path / "author_data.json") as f:
            self.author_data = json.load(f)
        self.author_data = json.loads(self.author_data)
        self.author_data_ids = set([int(author["id"]) for author in self.author_data])
        logger.info("Successfully loaded author_data.json!")

        end_time = datetime.now()
        logger.info("Duration of data loading: {}".format(end_time - start_time))

    # def load_dataset(self):
    #     logger.info("Loading authors.edgelist...")
    #     self.authors_graph = nx.read_edgelist(
    #         self.dataset_path / (self.dataset_name + "_" + "authors.edgelist"),
    #         create_using=nx.DiGraph,
    #     )
    #     logger.info("Successfully loaded authors.edgelist!")
    #
    #     logger.info("Loading papers.edgelist...")
    #     self.citation_graph = nx.read_edgelist(
    #         self.dataset_path / (self.dataset_name + "_" + "papers.edgelist"),
    #         create_using=nx.DiGraph,
    #     )
    #     logger.info("Successfully loaded papers.edgelist!")

    async def add_author(self, author_name: str, features_vector: List[int]) -> dict:
        max_id = max(self.author_data_ids)
        new_id = max_id + 1
        self.author_data_ids.add(new_id)
        new_author = {"name": author_name, "id": str(new_id)}
        self.authors_features.loc[new_id] = features_vector
        self.A.add_node(new_id)
        self.author_data.append(new_author)
        return new_author

    async def find_authors(self, search_text: str, threshold: int = 25) -> List[dict]:
        result_authors = []
        search_text = search_text.lower()
        search_text_len = len(search_text)
        for author in self.author_data:
            author_fullname: str = author["name"].lower()
            for author_part_name in author_fullname.split():
                matching_part: str = author_part_name[:search_text_len]
                if search_text == matching_part:
                    result_authors.append(author)
                    if len(result_authors) > threshold:
                        return result_authors
                    continue
        return result_authors

    async def find_author_feature_vector(self, author_id: int) -> List[int]:
        try:
            result_feature_vector = self.authors_features.loc[author_id].tolist()
        except Exception as e:
            logger.error(f"error : {e}")
            result_feature_vector = [0] * 19
        return result_feature_vector

    async def get_author(self, author_id: int = None, author_name: str = None) -> dict:
        result_author: dict = dict()
        if author_id:
            author_id_str: str = str(author_id)
            for author in self.author_data:
                current_author_id: str = author["id"]
                if author_id_str == current_author_id:
                    result_author = author
                    break
        elif author_name:
            author_name_lower = author_name.lower()
            for author in self.author_data:
                current_author_name: str = author["name"].lower()
                if author_name_lower == current_author_name:
                    result_author = author
                    break
        else:
            raise ValueError("Nothing was given")

        if result_author:
            return result_author
        else:
            raise ValueError("No author was found")
