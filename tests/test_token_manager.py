import pytest
from src.rag_pipeline import TokenManager


def test_TokenManager_calc_initial_tokens(rag_cfg):
    """
    - test calc available tokens at start of process with 10k total tokens 
    - _calc_initial_tokens triggered at init & assigns value to prop remaining_tokens
    """
    rag_cfg.total_available_tokens = 10000
    tm = TokenManager(rag_cfg)
    assert isinstance(tm._calc_initial_tokens(), int)
    assert tm._calc_initial_tokens() == 8200
    assert tm.remaining_tokens == 8200


def test_TokenManager_user_query_tokens(rag_cfg):
    """ getter user_query_tokens is derived from remaining_tokens * cfg user_query_share"""
    rag_cfg.total_available_tokens = 10000
    tm = TokenManager(rag_cfg)
    assert tm.user_query_tokens == 1640


def test_TokenManager_rag_context_tokens(rag_cfg):
    """ getter rag context tokens is derived from remaining tokens and cfg rag & llm share """
    rag_cfg.rag_content_share = 0.7
    rag_cfg.llm_response_share = 0.2
    tm = TokenManager(rag_cfg)
    tm.remaining_tokens = 7000
    assert tm.rag_context_tokens == 5444


def test_TokenManager_get_amount_tokens(tk_man, user_prompt):
    """
    - fixture user prompt has 14 words, must be more tokens
    - 18 tokens with tiktoken cl100k_base
    """
    token_amount = tk_man.get_token_amount(user_prompt)
    assert isinstance(token_amount, int) and token_amount == 18


def test_TokenManager_reduce_remaining_tokens_valid(rag_cfg, user_prompt):
    """
    - remaining_tokens after itit with 10k: 8200
    - user_prompt has 18 tokens
    """
    rag_cfg.total_available_tokens = 10000
    tm = TokenManager(rag_cfg)
    tm.reduce_remaining_tokens(user_prompt)
    assert tm.remaining_tokens == 8182


def test_TokenManager_reduce_remaining_tokens_invalid(rag_cfg, user_prompt):
    with pytest.raises(ValueError):
        rag_cfg.total_available_tokens = 10
        tm = TokenManager(rag_cfg)
        tm.reduce_remaining_tokens(user_prompt)


def test_TokenManager_reset_remaining_tokens(tk_man):
    """ default total tokens 6k; remaining at init 4680 """
    tk_man.remaining_tokens = 100000
    tk_man.reset_state()
    assert tk_man.remaining_tokens == 4680
