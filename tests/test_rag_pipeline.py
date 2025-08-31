import pytest
from src.rag_pipeline import RAGPipeline
import json




def test_RAGEngine_find_semantic_matches_top3(rag_eng, mock_user_prompt):
    """ top 3 articles must be part of rag_context independent of prompt / cosine similarity """
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_semantic_matches(mock_user_prompt)
    assert all(article in rag_eng.rag_context for article in ["Article 1", "Article 2", "Article 3"])


def test_RAGEngine_format_rag_context(rag_eng, art_1_final_rag_str):
    result = rag_eng.db.collections["articles"].get(ids=["article_001"])
    assert rag_eng._format_rag_context((result["documents"][0], 4.2)) == art_1_final_rag_str


def test_RAGEngine_reset_state(rag_eng):
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_direct_matches("Article 1")
    assert "Article 1" in rag_eng.rag_context
    rag_eng._reset_state()
    assert rag_eng.rag_context == "" and rag_eng.articles_relationships == [] \
           and rag_eng.used_ids == set()


def test_RAGEngine_direct_search_meta(rag_eng):
    # fill the rag_ops_tokens manually -> normally done by _execute...
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_direct_matches("Article 1")
    assert "Article 1" in rag_eng.rag_context
    print(rag_eng.articles_relationships)
    assert json.loads(rag_eng.articles_relationships[0].get("related_recitals")) == ["recital_001"]
    assert json.loads(rag_eng.articles_relationships[0].get("related_annexes")) == ["annex_01"]
    assert json.loads(rag_eng.articles_relationships[0].get("related_articles")) == ["article_002"]


def test_RAGEngine_direct_search_base(rag_eng):
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_direct_matches("find Article 1 for me!")
    assert "Article 1" in rag_eng.rag_context
    rag_eng._find_direct_matches("find article 1 for me!")
    assert "Article 1" in rag_eng.rag_context
    rag_eng._find_direct_matches("find Annex 3 for me!")
    assert "about high-risk AI systems" in rag_eng.rag_context
    rag_eng._find_direct_matches("find Recital 2 for me!")
    assert "union values" in rag_eng.rag_context
    # check for used ids
    assert "recital_002" in rag_eng.used_ids


def test_RAGPipeline_reset_state(rag_pipe, mock_user_prompt, mock_chatgroq, mock_rag_context):
    rag_pipe.rag_context = mock_rag_context
    rag_pipe.process_query(user_prompt=mock_user_prompt, rag_enriched=True)
    assert rag_pipe.rag_context != "" and rag_pipe.model is not None
    rag_pipe._reset_state()
    assert rag_pipe.rag_context == "" and rag_pipe.model is None


def test_RAGPipeline_process_query_base(rag_pipe, mock_user_prompt, mock_chatgroq):
    """ test with mock_chatgroq patch to mock groqchat instances to prevent real llm calls"""
    result = rag_pipe.process_query(user_prompt=mock_user_prompt, rag_enriched=False)
    assert result == "Mocked LLM response for testing!"


def test_RAGPipeline_validate_user_prompt_fail(rag_pipe, mock_user_prompt_too_long):
    with pytest.raises(AssertionError):
        rag_pipe._validate_user_prompt(mock_user_prompt_too_long)


def test_RAGPipeline_validate_user_prompt(rag_pipe, mock_user_prompt):
    """ check val of user prompt within valid range """
    assert rag_pipe._validate_user_prompt(mock_user_prompt)


def test_RAGPipeline_init_base(rag_eng, rag_cfg, tk_man):
    """ test init obj & user_query_len getter"""
    pipe = RAGPipeline(
        config=rag_cfg,
        tm=tk_man,
        rag_engine=rag_eng
    )
    assert pipe.rag_context == "" and pipe.model is None
    assert pipe.user_query_len == 3744
