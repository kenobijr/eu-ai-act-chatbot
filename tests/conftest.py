import pytest
from unittest.mock import MagicMock, patch
from src.config import RAGConfig, DBConfig
from src.rag_pipeline import TokenManager, RAGEngine, RAGPipeline
from src.vector_db import DB



@pytest.fixture
def mock_chatgroq():
    with patch('src.rag_pipeline.ChatGroq') as mock_groq:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value.content = "Mocked LLM response for testing!"
        mock_groq.return_value = mock_instance
        yield mock_groq


@pytest.fixture
def rag_pipe(rag_cfg, tk_man, rag_eng):
    return RAGPipeline(
        config=rag_cfg,
        tm=tk_man,
        rag_engine=rag_eng,
    )


@pytest.fixture
def rag_eng(rag_cfg, tk_man, mock_db):
    return RAGEngine(
        config=rag_cfg,
        tm=tk_man,
        db=mock_db,
    )


@pytest.fixture
def tk_man(rag_cfg):
    return TokenManager(
        config=rag_cfg,
    )


@pytest.fixture
def rag_cfg(mock_rag_disabled_systemmessage, mock_rag_enabled_systemmessage):
    cfg = RAGConfig()
    cfg.system_message_rag_disabled = mock_rag_disabled_systemmessage
    cfg.system_message_rag_enabled = mock_rag_enabled_systemmessage
    return cfg


@pytest.fixture
def art_1_final_rag_str():

    return """[Relevance: -320%]
Test Article 1: Subject Matter
            Part of Chapter I: General Provisions hehehe
            Date of entry into force: 2 February 2025
            Mock content for article 1 about subject matter

---

"""


@pytest.fixture
def mock_rag_context():
    return """[Relevance: 52%]
Article 58: Detailed Arrangements for AI Regulatory Sandboxes
Mock content about AI regulatory sandboxes arrangements and procedures.

---

[Relevance: 44%]
Article 1: Subject Matter
Mock content about regulation purpose and internal market functioning.

---

[Relevance: 40%]
Article 62: Measures for Providers and Deployers, in Particular SMEs
Mock content about SME support measures and priority access.

---

[Relevance: 62%]
Recital 139
Mock recital about AI regulatory sandbox objectives and innovation fostering.

---

[Relevance: 49%]
Definition: AI regulatory sandbox
Mock definition of controlled framework for AI system development and testing.

---

[Relevance: 45%]
Recital 138
Mock recital about AI development requirements and regulatory oversight.

---

"""


@pytest.fixture
def mock_user_prompt():
    return "How do the requirements for AI regulatory sandboxes \
            relate to innovation support for SMEs?"


@pytest.fixture
def mock_user_prompt_too_long():
    """ create a user prompt that exceeds the token limit of 936 tokens """
    base_question = "How do the requirements for AI regulatory sandboxes relate to innovation \
                    support for SMEs?"
    return base_question * 60


@pytest.fixture
def mock_db(mock_entity_jsons, mock_db_cfg):
    """ mock DB to test DB read mode """
    return DB.build_mode(config=mock_db_cfg)


@pytest.fixture
def shared_tmp_path(tmp_path):
    """ shared path for mock db config & mock jsons for DB build mode """
    return tmp_path


@pytest.fixture
def mock_db_cfg(shared_tmp_path):
    """ database config fixture for testing DB build mode """
    return DBConfig(
        entities=["annexes", "articles", "definitions", "recitals"],
        data_dir=shared_tmp_path / "data" / "raw",
        file_extension=".json",
        save_dir=shared_tmp_path / "data" / "chroma_db"
    )


@pytest.fixture
def mock_entity_jsons(shared_tmp_path):
    """create mock json data for all entities for DB build mode"""
    save_dir = shared_tmp_path / "data" / "raw"
    save_dir.mkdir(parents=True, exist_ok=True)
    # mock data for each entity with 3 entries each
    mock_data = {
        "annexes": {
            "annex_01": {
                "id": "annex_01",
                "title": "Test Annex I: Harmonisation Legislation",
                "text_content": "Mock content for annex 1 about harmonisation legislation"
            },
            "annex_02": {
                "id": "annex_02", 
                "title": "Test Annex II: Criminal Offences",
                "text_content": "Mock content for annex 2 about criminal offences"
            },
            "annex_03": {
                "id": "annex_03",
                "title": "Test Annex III: High-Risk AI Systems", 
                "text_content": "Mock content for annex 3 about high-risk AI systems"
            }
        },
        "articles": {
            "article_001": {
                "id": "article_001",
                "title": "Test Article 1: Subject Matter",
                "text_content": "Mock content for article 1 about subject matter",
                "chapter_title": "Chapter I: General Provisions",
                "section_title": "hehehe",
                "entry_date": "2 February 2025",
                "related_recital_ids": ["recital_001"],
                "related_annex_ids": ["annex_01"],
                "related_article_ids": ["article_002"]
            },
            "article_002": {
                "id": "article_002",
                "title": "Test Article 2: Scope",
                "text_content": "Mock content for article 2 about scope",
                "chapter_title": "Chapter I: General Provisions", 
                "section_title": None,
                "entry_date": "2 February 2025",
                "related_recital_ids": ["recital_002"],
                "related_annex_ids": ["annex_02"],
                "related_article_ids": ["article_001", "article_003"]
            },
            "article_003": {
                "id": "article_003",
                "title": "Test Article 3: Definitions",
                "text_content": "Mock content for article 3 about definitions",
                "chapter_title": "Chapter I: General Provisions",
                "section_title": None, 
                "entry_date": "2 February 2025",
                "related_recital_ids": ["recital_003"],
                "related_annex_ids": [],
                "related_article_ids": ["article_002"]
            }
        },
        "definitions": {
            "definition_01": {
                "id": "definition_01",
                "title": "AI system",
                "text_content": "Mock definition for AI system"
            },
            "definition_02": {
                "id": "definition_02",
                "title": "risk",
                "text_content": "Mock definition for risk"
            },
            "definition_03": {
                "id": "definition_03",
                "title": "provider",
                "text_content": "Mock definition for provider"
            }
        },
        "recitals": {
            "recital_001": {
                "id": "recital_001",
                "text_content": "Mock recital 1 about regulation purpose"
            },
            "recital_002": {
                "id": "recital_002",
                "text_content": "Mock recital 2 about union values"
            },
            "recital_003": {
                "id": "recital_003", 
                "text_content": "Mock recital 3 about AI system deployment"
            }
        }
    }
    # save json files
    import json
    for entity_type, entities in mock_data.items():
        file_path = save_dir / f"{entity_type}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)
    return save_dir


@pytest.fixture
def mock_rag_disabled_systemmessage():
    return "Mock system message for RAG disabled mode - legal expert providing AI Act guidance."


@pytest.fixture
def mock_rag_enabled_systemmessage():
    return "Mock system message for RAG enabled mode - legal expert with document context."
