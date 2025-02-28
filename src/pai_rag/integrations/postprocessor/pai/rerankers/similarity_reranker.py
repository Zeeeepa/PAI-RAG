from typing import List
from llama_index.core.schema import NodeWithScore


class SimilarityReranker:
    def __init__(
        self,
        similarity_threshold: float = 0,
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold

    async def arerank(self, query_str: str, nodes: List[NodeWithScore]):
        return [node for node in nodes if node.score > self.similarity_threshold]
