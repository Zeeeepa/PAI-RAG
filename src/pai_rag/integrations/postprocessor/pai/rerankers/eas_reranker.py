from typing import List
import time
import httpx
from llama_index.core.schema import NodeWithScore
from loguru import logger


MAX_INPUT_LENGTH = 120


class EasReranker:
    def __init__(
        self,
        endpoint: str,
        token: str,
        top_n: int,
        similarity_threshold: float = 0,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.token = token
        self.top_n = top_n
        self.rerank_capacity = 20  # maximum rerank 20 nodes
        self.similarity_threshold = similarity_threshold

    async def _arerank(self, query: str, texts: List[str]):
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            payload = {
                "query": query,
                "texts": texts,
            }
            response = await client.post(
                self.endpoint,
                headers={"Authorization": self.token},
                json=payload,
            )

            item_scores = response.json()
            logger.info(
                f"Finished query reranking api with {payload}. Elapsed: {time.time() - start_time} seconds."
            )

            item_scores = [
                item
                for item in item_scores
                if item["score"] >= self.similarity_threshold
            ]
            sorted_item_scores = sorted(
                item_scores, key=lambda x: x["score"], reverse=True
            )

            return [item["index"] for item in sorted_item_scores]

    async def arerank(self, query_str: str, nodes: List[NodeWithScore]):
        candidate_nodes = []
        seen_texts = set()
        texts = []
        for node in nodes:
            node_text = node.node.get_text()[:MAX_INPUT_LENGTH]
            print("====", node_text)
            if node_text not in seen_texts:
                candidate_nodes.append(node)
                texts.append(node_text)
                if len(candidate_nodes) >= self.rerank_capacity:
                    break
                seen_texts.add(node_text)

        if len(texts) <= self.top_n:
            return candidate_nodes

        sorted_indexes = await self._arerank(query=query_str, texts=texts)
        return [candidate_nodes[i] for i in sorted_indexes[: self.top_n]]
