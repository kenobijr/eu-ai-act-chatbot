from src.rag_pipeline import TokenManager
from src.config import RAGConfig


def test_TokenManager__calc_effective_tokens():
    tm = TokenManager(RAGConfig())
    assert isinstance(tm._calc_initial_tokens(), int)
    assert tm._calc_initial_tokens() == 4680
