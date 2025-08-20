import pytest
from src.config import RAGConfig
from src.rag_pipeline import TokenManager, RAGEngine, RAGPipeline


# @pytest.fixture
# def rag_pipe(rag_cfg, tk_man, rag_eng):
#     return RAGPipeline(config=rag_cfg, tm=tk_man, rag_engine=rag_eng)


# @pytest.fixture
# def rag_eng(tk_man, rag_cfg):
#     return RAGEngine(
#         config=rag_cfg,
#         tm=tk_man,
#     )


@pytest.fixture
def tk_man(rag_cfg):
    return TokenManager(
        config=rag_cfg,
    )


@pytest.fixture
def rag_cfg():
    return RAGConfig()


@pytest.fixture
def user_prompt():
    return "How do the requirements for AI regulatory sandboxes \
            relate to innovation support for SMEs?"