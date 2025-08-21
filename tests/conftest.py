import pytest
from src.config import RAGConfig, DBConfig
from src.rag_pipeline import TokenManager, RAGEngine, RAGPipeline
from src.vector_db import DB




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


@pytest.fixture
def mock_db(mock_entity_jsons, mock_db_cfg):
    """ mock DB to test DB read mode """
    return DB.build_mode(config=mock_db_cfg)


@pytest.fixture
def shared_tmp_path(tmp_path):
    """ shared path for mock db config & mock jsons for DB build mode"""
    return tmp_path


@pytest.fixture
def mock_db_cfg(shared_tmp_path):
    """database config fixture for testing DB build mode"""
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
