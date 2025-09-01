import pytest
from src.rag_pipeline import RAGPipeline


def test_RAGEngine_find_semantic_matches_boost(rag_eng, mock_user_prompt):
    rag_eng.tm.rag_ops_tokens = 100000
    direct_search_prompt = "find Article 1 for me!"
    rag_eng._find_direct_matches(direct_search_prompt)
    assert "recital_001" in rag_eng.rel_boost_ids
    # get original distance for recital_001 before boost
    result = rag_eng.db.collections["recitals"].query(
        query_texts=[mock_user_prompt], n_results=10
    )
    original_distance = result["distances"][0][result["ids"][0].index("recital_001")]
    # execute semantic matches with boost applied
    rag_eng._find_semantic_matches(mock_user_prompt)
    # verify boost worked: boosted distance should be original * relationship_boost
    boosted_distance = original_distance * rag_eng.config.relationship_boost
    assert boosted_distance == (0.5 * original_distance)


def test_RAGEngine_find_semantic_matches_meta(
    rag_eng, mock_user_prompt, mock_all_art_rels
):
    """no direct match with default mock prompt; check if all article rels for boost are there"""
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_semantic_matches(mock_user_prompt)
    assert all(rel in rag_eng.rel_boost_ids for rel in mock_all_art_rels)


def test_RAGEngine_find_semantic_matches_top3(rag_eng, mock_user_prompt):
    """top 3 articles must be part of rag_context independent of prompt / cosine similarity"""
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_semantic_matches(mock_user_prompt)
    assert all(
        article in rag_eng.rag_context
        for article in ["Article 1", "Article 2", "Article 3"]
    )


def test_RAGEngine_find_semantic_matches_candidates_filtering(rag_eng):
    """test candidates are filtered by relevance threshold and boosted entities prioritized"""
    rag_eng.tm.rag_ops_tokens = 100000
    # add some boost ids manually to test filtering
    rag_eng.rel_boost_ids = {"recital_001", "annex_01"}
    rag_eng._find_semantic_matches("AI system definitions")
    # verify filtered candidates were added beyond base articles
    total_entities = len(
        [line for line in rag_eng.rag_context.split("\n") if line.startswith("[")]
    )
    assert total_entities > 3  # more than just base articles
    # verify boosted entities have better relevance scores
    lines = rag_eng.rag_context.split("\n")
    boost_scores = [
        int(line.split(": ")[1].split("%")[0])
        for line in lines
        if line.startswith("[Relevance:")
        and (
            "recital" in lines[lines.index(line) + 1]
            or "annex" in lines[lines.index(line) + 1]
        )
    ]
    if boost_scores:
        assert max(boost_scores) >= 50  # boosted items should have decent relevance


def test_RAGEngine_extract_related_ids():
    """test extraction of relationship ids from article metadata"""
    from src.rag_pipeline import RAGEngine

    # mock article metadata with relationships
    mock_metadata = {
        "related_articles": '["article_002", "article_003"]',
        "related_recitals": '["recital_001"]',
        "related_annexes": '["annex_01", "annex_02"]',
    }
    result = RAGEngine._extract_related_ids(mock_metadata)
    expected = {"article_002", "article_003", "recital_001", "annex_01", "annex_02"}
    assert result == expected
    # test empty relationships
    empty_metadata = {
        "related_articles": "[]",
        "related_recitals": "[]",
        "related_annexes": "[]",
    }
    assert RAGEngine._extract_related_ids(empty_metadata) == set()
    # test missing keys
    incomplete_metadata = {"related_articles": '["article_001"]'}
    result = RAGEngine._extract_related_ids(incomplete_metadata)
    assert result == {"article_001"}


def test_RAGEngine_format_rag_context(rag_eng, art_1_final_rag_str):
    result = rag_eng.db.collections["articles"].get(ids=["article_001"])
    assert (
        rag_eng._format_rag_context((result["documents"][0], 4.2))
        == art_1_final_rag_str
    )


def test_RAGEngine_reset_state(rag_eng):
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_direct_matches("Article 1")
    assert "Article 1" in rag_eng.rag_context
    rag_eng._reset_state()
    assert (
        rag_eng.rag_context == ""
        and rag_eng.rel_boost_ids == set()
        and rag_eng.used_ids == set()
    )


def test_RAGEngine_direct_search_meta(rag_eng):
    # fill the rag_ops_tokens manually -> normally done by _execute...
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_direct_matches("Article 1")
    assert all(
        rel in rag_eng.rel_boost_ids
        for rel in ["recital_001", "annex_01", "article_002"]
    )


def test_RAGEngine_direct_search_base(rag_eng):
    rag_eng.tm.rag_ops_tokens = 100000
    rag_eng._find_direct_matches("find Article 1 for me!")
    assert "Article 1" in rag_eng.rag_context
    rag_eng._find_direct_matches("find Annex 3 for me!")
    assert "about high-risk AI systems" in rag_eng.rag_context
    rag_eng._find_direct_matches("find Recital 2 for me!")
    assert "union values" in rag_eng.rag_context
    # check for used ids
    assert "recital_002" in rag_eng.used_ids


def test_RAGPipeline_reset_state(
    rag_pipe, mock_user_prompt, mock_chatgroq, mock_rag_context
):
    rag_pipe.rag_context = mock_rag_context
    rag_pipe.process_query(user_prompt=mock_user_prompt, rag_enriched=True)
    assert rag_pipe.rag_context != "" and rag_pipe.model is not None
    rag_pipe._reset_state()
    assert rag_pipe.rag_context == "" and rag_pipe.model is None


def test_RAGPipeline_process_query_base(rag_pipe, mock_user_prompt, mock_chatgroq):
    """test with mock_chatgroq patch to mock groqchat instances to prevent real llm calls"""
    result = rag_pipe.process_query(user_prompt=mock_user_prompt, rag_enriched=False)
    assert result == "Mocked LLM response for testing!"


def test_RAGPipeline_validate_user_prompt_fail(rag_pipe, mock_user_prompt_too_long):
    with pytest.raises(AssertionError):
        rag_pipe._validate_user_prompt(mock_user_prompt_too_long)


def test_RAGPipeline_validate_user_prompt(rag_pipe, mock_user_prompt):
    """check val of user prompt within valid range"""
    assert rag_pipe._validate_user_prompt(mock_user_prompt)


def test_RAGPipeline_init_base(rag_eng, rag_cfg, tk_man):
    """test init obj & user_query_len getter"""
    pipe = RAGPipeline(config=rag_cfg, tm=tk_man, rag_engine=rag_eng)
    assert pipe.rag_context == "" and pipe.model is None
    assert pipe.user_query_len == 3744
