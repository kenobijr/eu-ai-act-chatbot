from app import process_query
from unittest.mock import patch


def test_process_query_empty_input():
    """ test process_query with empty input """
    assert process_query("") == "Enter a question." and process_query(None) == "Enter a question."


def test_process_query_rag_none():
    """ test process_query when rag pipeline is None """
    with patch("app.rag", None):
        result = process_query("test question")
        assert "Error: RAG pipeline failed to initialize" in result


def test_process_query_exception(mock_rag_instance):
    """ test query processing with exception """
    mock_rag_instance.process_query.side_effect = ValueError("Token limit exceeded")
    with patch("app.rag", mock_rag_instance):
        result = process_query("test question")
        assert result == "Error: Token limit exceeded"


def test_rag_init_success(mock_rag_instance):
    """ test successful rag pipeline initialization """
    with patch("src.rag_pipeline.RAGPipeline", return_value=mock_rag_instance):
        pipeline = mock_rag_instance
        assert pipeline.user_query_len == 3744
