from enum import Enum
from typing import List, Literal, Union
from pydantic import BaseModel
from llama_index.core.bridge.pydantic import PrivateAttr
import os
from pai_rag.integrations.postprocessor.pai.rerankers.eas_reranker import EasReranker
from pai_rag.integrations.postprocessor.pai.rerankers.local_reranker import (
    LocalModelReranker,
)
from pai_rag.integrations.postprocessor.pai.rerankers.similarity_reranker import (
    SimilarityReranker,
)
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from loguru import logger


# rerank constants
DEFAULT_RERANK_MODEL = "bge-reranker-base"
DEFAULT_SIMILARITY_THRESHOLD = 0
DEFAULT_RERANK_SIMILARITY_THRESHOLD = 0
DEFAULT_RERANK_TOP_N = 5


class PostProcessorType(str, Enum):
    no_reranker = "no-reranker"
    reranker_model = "model-based-reranker"
    eas_reranker_api = "eas-reranker-api"


class BasePostProcessorConfig(BaseModel):
    reranker_type: Literal[
        PostProcessorType.no_reranker
    ] = PostProcessorType.no_reranker


class SimilarityPostProcessorConfig(BasePostProcessorConfig):
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD


class RerankModelPostProcessorConfig(BasePostProcessorConfig):
    reranker_type: Literal[
        PostProcessorType.reranker_model
    ] = PostProcessorType.reranker_model
    reranker_model: str = DEFAULT_RERANK_MODEL
    top_n: int = DEFAULT_RERANK_TOP_N
    similarity_threshold: float = DEFAULT_RERANK_SIMILARITY_THRESHOLD


# https://pai.console.aliyun.com/?regionId=cn-hangzhou&scm=20140722.S_learn._.ID_learn-RL_PAI-LOC_console_console-OR_ser-V_4-P0_0&spm=5176.12818093_47.console-base_search-panel.dtab-product_learn.3be916d0kVdU78&workspaceId=146707#/quick-start/models/bge-reranker-large/intro
class EasRerankerPostProcessorConfig(BasePostProcessorConfig):
    reranker_type: Literal[
        PostProcessorType.eas_reranker_api
    ] = PostProcessorType.eas_reranker_api
    endpoint: str = ""
    token: str = ""
    top_n: int = DEFAULT_RERANK_TOP_N
    similarity_threshold: float = DEFAULT_RERANK_SIMILARITY_THRESHOLD


def create_postprocessors(config: BasePostProcessorConfig):
    if isinstance(config, RerankModelPostProcessorConfig):
        pai_model_dir = os.getenv("PAI_RAG_MODEL_DIR", "./model_repository")
        model = os.path.join(pai_model_dir, config.reranker_model)
        reranker = LocalModelReranker(
            model=model,
            top_n=config.top_n,
            similarity_threshold=config.similarity_threshold,
            callback_manager=Settings.callback_manager,
        )
        logger.info(
            f"""[Reranker]: Reranker model inited
                model_name: {model}
                top_n: {config.top_n},
                similarity_threshold: {config.similarity_threshold}"""
        )
    elif isinstance(config, SimilarityPostProcessorConfig):
        reranker = SimilarityReranker(
            similarity_threshold=config.similarity_threshold,
        )
        logger.info(
            f"""[Reranker]: Similarity reranker inited
                similarity_threshold: {config.similarity_threshold}"""
        )
    elif isinstance(config, EasRerankerPostProcessorConfig):
        reranker = EasReranker(
            endpoint=config.endpoint,
            token=config.token,
            top_n=config.top_n,
            similarity_threshold=config.similarity_threshold,
        )
        logger.info(
            f"""[Reranker]: EAS Reranker inited
                Endpoint: {config.endpoint}
                Token: {config.token},
                similarity_threshold: {config.similarity_threshold}"""
        )

    return reranker


class PaiPostProcessor:
    _reranker: Union[
        LocalModelReranker, SimilarityReranker, EasReranker
    ] = PrivateAttr()

    def __init__(
        self,
        postprocessor_config: BasePostProcessorConfig,
    ):
        super().__init__()
        self._reranker = create_postprocessors(postprocessor_config)

    @classmethod
    def class_name(cls) -> str:
        return "PaiPostProcessor"

    async def arerank(self, query_str: str, nodes: List[NodeWithScore]):
        return await self._reranker.arerank(query_str, nodes)
