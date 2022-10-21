import pathlib
from typing import List, Tuple, Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from .models import (
    AuthorEdgeList,
    NewAuthor,
    AuthorEdge,
    ModelPredictResultResponse,
    Author,
    DefaultMessageResponse,
    AuthorIDs,
    ListFeatureVector,
    AuthorFeatureVector,
)
from ..core.settings import Settings
import app.ml as ml


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.current_queue: List[Tuple[WebSocket, Any]] = []
        self.current_status_free: bool = True

    async def connect(self, websocket: WebSocket):
        logger.debug("Now in connect.")
        await websocket.accept()
        self.active_connections.append(websocket)
        try:
            message = DefaultMessageResponse(
                data="queue_message",
                comment="You have successfully connected to the websocket!",
            )
            await websocket.send_json(message.json())
        except WebSocketDisconnect as e:
            logger.error(f"error : {e}")
            await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket):
        logger.debug("Now in disconnect.")
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"error : {e}")
        await self.dequeue(websocket)

    async def update_current_status(self):
        logger.debug(f"Now updating status.")
        for i, x in enumerate(self.current_queue):
            logger.debug(f"In other queue.")
            # logger.debug(f"second ws : {websocket}")
            if i == 0:
                continue
            else:
                logger.debug(f"Updating queue messaging status.")
                websocket = x[0]
                # logger.debug("It should start sending text now!")
                try:
                    message = DefaultMessageResponse(
                        data="queue_message",
                        comment=f"Your place in queue: {i}. Please wait!",
                    )
                    await websocket.send_json(message.json())
                except WebSocketDisconnect as e:
                    logger.error(f"error : {e}")
                    await self.disconnect(websocket)
        if len(self.current_queue) >= 1 and self.current_status_free:
            x = self.current_queue[0]
            websocket = x[0]

            try:
                message = DefaultMessageResponse(
                    data="queue_message",
                    comment="Your queue is up! Model will now prepare your results!",
                )
                await websocket.send_json(message.json())
            except Exception as e:
                logger.error(f"error : {e}")
                await self.disconnect(websocket)

            # logger.debug("It should send text!")
            try:
                # logger.debug("It should start predicting now!")
                await self.predict(x[1], websocket)
            except Exception as e:
                logger.error(f"error : {e}")
                self.current_status_free = True
                await self.disconnect(websocket)

    async def queue(self, data, websocket: WebSocket):
        logger.debug(f"In queue now.")
        ws_queue_idx = await self.find_ws_queue(websocket)
        if ws_queue_idx == -1:
            logger.debug(f"New queue.")
            self.current_queue.append((websocket, data))
            logger.debug(f"Added to current queue.")
            message = DefaultMessageResponse(
                data="queue_message", comment="We have saved your input data.."
            )
            await websocket.send_json(message.json())
            logger.debug(f"Sent message to websocket about save.")
            await self.update_current_status()
        else:
            if ws_queue_idx != 0:
                logger.debug(f"Update queue.")
                self.current_queue[ws_queue_idx] = (websocket, data)
                message = DefaultMessageResponse(
                    data="queue_message", comment="We have updated your input data."
                )
                await websocket.send_json(message.json())
                logger.debug(f"Sent message to websocket about update.")

    async def find_ws_queue(self, websocket: WebSocket):
        logger.debug(f"In find_ws_queue now.")
        queue_index: int = -1
        for i, x in enumerate(self.current_queue):
            if x[0] == websocket:
                queue_index = i
                break
        return queue_index

    async def dequeue(self, websocket: WebSocket):
        logger.debug("Now in dequeue.")
        queue_index = await self.find_ws_queue(websocket)
        if queue_index != -1:
            self.current_queue.pop(queue_index)
            await self.update_current_status()

    async def predict(self, data: dict, websocket: WebSocket):
        self.current_status_free = False
        if "threshold" not in data:
            data["threshold"] = 0.75
        # logger.debug("It should start sending text now!")
        message = DefaultMessageResponse(data="predict_status", comment="0")
        await websocket.send_json(message.json())

        target_authors = await run_in_threadpool(inference.get_authors, data["authors"])
        (
            train_data,
            data_citation,
            authors_to_papers,
        ) = await run_in_threadpool(inference.get_train_data)

        # logger.debug("It should start sending text now!")
        message = DefaultMessageResponse(data="predict_status", comment="1")
        await websocket.send_json(message.json())

        test_data = await run_in_threadpool(
            inference.get_test_data, target_authors, train_data
        )
        (
            result,
            result_z,
            result_z_sjr,
            result_z_hi,
            result_z_ifact,
            result_z_numb,
        ) = await run_in_threadpool(
            inference.predict,
            train_data,
            test_data,
            data_citation,
            authors_to_papers,
            data["threshold"],
        )

        try:
            logger.debug("It should start sending text now!")
            message = DefaultMessageResponse(data="predict_status", comment="2")
            await websocket.send_json(message.json())
        except Exception as e:
            logger.error(e)

        try:
            if len(result) == 0:
                message = DefaultMessageResponse(
                    data="predict_status",
                    comment="Error. No graph was constructed, or rather, it is empty.",
                )
                await websocket.send_json(message.json())
                logger.debug(f"Sent message to websocket about emtpy graph.")
        except Exception as e:
            logger.error(e)

        try:
            logger.debug("It should start sending text now!")
            message = DefaultMessageResponse(data="predict_status", comment="3")
            await websocket.send_json(message.json())
        except Exception as e:
            logger.error(e)

        result_edges = [AuthorEdge(author_a=x[0][0], author_b=x[0][1]) for x in result]
        result_edgelist = AuthorEdgeList(data=result_edges)

        result_response = ModelPredictResultResponse(
            author_edgelist=result_edgelist,
            z=result_z,
            z_sjr=result_z_sjr,
            z_hi=result_z_hi,
            z_ifact=result_z_ifact,
            z_numb=result_z_numb,
        )

        logger.debug(f"result_edges : {result_edgelist}")
        logger.debug(f"result_response : {result_response}")

        logger.debug("It should start sending text now!")
        message = DefaultMessageResponse(data="predict_status", comment="4")
        await websocket.send_json(message.json())

        # logger.debug("It should send result now!")
        await websocket.send_json(result_response.json())
        self.current_status_free = True
        await self.disconnect(websocket)


app = FastAPI(
    title=Settings().project_name,
    version=Settings().version,
    description=Settings().fast_api_description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = pathlib.Path(__file__).parent.resolve().parent.parent
inference: ml.inference.GraphInference = ml.inference.GraphInference(
    root_path=BASE / "data"
)
manager: ConnectionManager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    logger.info(f"Docs: https://{Settings().hostname}:{Settings().port}/redoc")


@app.websocket("/ws_predict")
async def ws_predict(websocket: WebSocket):
    # logger.debug(f"first ws : {websocket}")
    await manager.connect(websocket)

    while True:
        try:
            data = await websocket.receive_json()
            # logger.debug(f"data : {data}")
            logger.debug(f"Got data {data}, now queue!")
            await manager.queue(data, websocket)

        except Exception as e:
            logger.error(f"error : {e}")
            break


@app.post("/add_author", response_model=Author)
async def add_author(new_author: NewAuthor):
    result_author = await inference.data.add_author(
        new_author.name, new_author.feature_vector
    )
    return Author(name=result_author["name"], id=result_author["id"])


@app.post("/get_author_feature_vector", response_model=ListFeatureVector)
async def get_author_feature_vector(authors: AuthorIDs):
    result_response = ListFeatureVector()
    for author in authors.authors:
        feature_vector = await inference.data.find_author_feature_vector(author)
        author_dict = await inference.data.get_author(author)
        author_response_object = AuthorFeatureVector(
            author=Author(name=author_dict["name"], id=author_dict["id"]),
            feature_vector=feature_vector,
        )
        result_response.data.append(author_response_object)
    return result_response


@app.websocket("/ws_search_authors")
async def websocket_authors(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            if len(data) != 0:
                # logger.debug(f"data : {data}")
                found_authors = await inference.data.find_authors(data)
                if len(found_authors) != 0:
                    # logger.debug(f"found_authors : {found_authors}")
                    await websocket.send_json(found_authors)

        except Exception as e:
            logger.error(f"error : {e}")
            break
    try:
        await websocket.close()
    except Exception as e:
        logger.error(f"error : {e}")
