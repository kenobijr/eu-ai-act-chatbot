import pytest
from src.rag_pipeline import TokenManager
from src.config import RAGConfig
import json


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
