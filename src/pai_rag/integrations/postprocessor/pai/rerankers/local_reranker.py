from typing import Any, List

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import MetadataMode, NodeWithScore


# TODO : Remove this
class LocalModelReranker:
    """Flag Embedding Reranker."""

    model: str = Field(description="BAAI Reranker model name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    similarity_threshold: float = Field(
        default=None, description="Similarity threshold for the reranker scores."
    )
    use_fp16: bool = Field(description="Whether to use fp16 for inference.")
    _model: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        similarity_threshold: None = None,
        model: str = "BAAI/bge-reranker-large",
        use_fp16: bool = False,
    ):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "Cannot import FlagReranker package, please install it: ",
                "pip install git+https://github.com/FlagOpen/FlagEmbedding.git",
            )

        self.top_n = top_n
        self.similarity_threshold = similarity_threshold
        self.model = model
        self.use_fp16 = use_fp16

        self._model = FlagReranker(
            model,
            use_fp16=use_fp16,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LocalModelReranker"

    def __repr__(self):
        return f"LocalModelReranker: top_n={self.top_n}, model={self.model}, similarity_threshold={self.similarity_threshold})"

    # very slow in fastapi
    async def arerank(
        self,
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[NodeWithScore]:
        if len(nodes) == 0 or len(nodes) <= self.top_n:
            return nodes

        query_and_nodes = [
            (
                query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]

        scores = self._model.compute_score(query_and_nodes)
        # a single node passed into compute_score returns a float
        if isinstance(scores, float):
            scores = [scores]

        assert len(scores) == len(nodes)

        for node, score in zip(nodes, scores):
            node.score = score

        new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
            : self.top_n
        ]

        if self.similarity_threshold is not None:
            new_nodes = [
                node for node in new_nodes if node.score > self.similarity_threshold
            ]

        return new_nodes
