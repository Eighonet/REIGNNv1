from typing import List, Any
from pydantic import BaseModel


class DefaultMessageResponse(BaseModel):
    data: str = "data"
    comment: str = "comment"


class Author(BaseModel):
    name: str = "Author"
    id: str = "101"


class NewAuthor(BaseModel):
    name: str = "Author"
    feature_vector: List[int] = []


class AuthorList(BaseModel):
    authors: List[Author] = []


class AuthorIDs(BaseModel):
    authors: List[int] = list()


class AuthorEdge(BaseModel):
    author_a: int = 0
    author_b: int = 1


class AuthorEdgeList(BaseModel):
    data: List[AuthorEdge] = list()


class ModelPredictResultResponse(BaseModel):
    author_edgelist: AuthorEdgeList = AuthorEdgeList()
    z: list = list()
    z_sjr: list = list()
    z_hi: list = list()
    z_ifact: list = list()
    z_numb: list = list()


class AuthorFeatureVector(BaseModel):
    author: Author = Author()
    feature_vector: List[int] = []


class ListFeatureVector(BaseModel):
    data: List[AuthorFeatureVector] = []
