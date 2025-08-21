from src.config import RAGConfig
from src.vector_db import DB
import pytest
import json


def test_DB_read_article(mock_db, mock_db_cfg, user_prompt):
    db = mock_db.read_mode(config=mock_db_cfg)
    result = db.collections["articles"].query(
        query_texts=[user_prompt],
        n_results=3,
    )
    # assertions on the overall result structure
    assert len(result["ids"][0]) == 3
    assert isinstance(result["distances"][0][0], float)
    # find the index of a specific article and then check its metadata
    article_001_index = result["ids"][0].index("article_001")
    article_002_index = result["ids"][0].index("article_002")
    article_003_index = result["ids"][0].index("article_003")
    # check the metadata for 'article_001'
    metadata_001 = result["metadatas"][0][article_001_index]
    assert metadata_001["type"] == "article"
    assert metadata_001["term"] == "Test Article 1: Subject Matter"
    assert metadata_001["chapter"] == "Chapter I: General Provisions"
    assert metadata_001["entry_date"] == "2 February 2025"
    assert json.loads(metadata_001.get("related_recitals")) == ["recital_001"]
    # check the metadata for 'article_002'
    metadata_002 = result["metadatas"][0][article_002_index]
    assert metadata_002["type"] == "article"
    assert metadata_002["term"] == "Test Article 2: Scope"
    assert json.loads(metadata_002.get("related_recitals")) == ["recital_002"]
    # check the metadata for 'article_003'
    metadata_003 = result["metadatas"][0][article_003_index]
    assert metadata_003["type"] == "article"
    assert metadata_003["term"] == "Test Article 3: Definitions"
    assert json.loads(metadata_003.get("related_recitals")) == ["recital_003"]
    assert metadata_003["chapter"] == "Chapter I: General Provisions"


def test_DB_read_annex(mock_db, mock_db_cfg, user_prompt):
    db = mock_db.read_mode(config=mock_db_cfg)
    result = db.collections["annexes"].query(
        query_texts=[user_prompt],
        n_results=3,
    )
    assert len(result["ids"][0]) == 3
    assert isinstance(result["distances"][0][0], float)
    annex_01_index = result["ids"][0].index("annex_01")
    annex_02_index = result["ids"][0].index("annex_02")
    metadata_01 = result["metadatas"][0][annex_01_index]
    assert metadata_01["type"] == "annex"
    assert metadata_01["term"] == "Test Annex I: Harmonisation Legislation"
    metadata_02 = result["metadatas"][0][annex_02_index]
    assert metadata_02["type"] == "annex"
    assert metadata_02["term"] == "Test Annex II: Criminal Offences"


def test_DB_read_definition(mock_db, mock_db_cfg, user_prompt):
    db = mock_db.read_mode(config=mock_db_cfg)
    result = db.collections["definitions"].query(
        query_texts=[user_prompt],
        n_results=3,
    )
    assert len(result["ids"][0]) == 3
    assert isinstance(result["distances"][0][0], float)
    def_01_index = result["ids"][0].index("definition_01")
    def_02_index = result["ids"][0].index("definition_02")
    metadata_01 = result["metadatas"][0][def_01_index]
    assert metadata_01["type"] == "definition"
    assert metadata_01["term"] == "AI system"
    metadata_02 = result["metadatas"][0][def_02_index]
    assert metadata_02["type"] == "definition"
    assert metadata_02["term"] == "risk"


def test_DB_read_recital(mock_db, mock_db_cfg, user_prompt):
    db = mock_db.read_mode(config=mock_db_cfg)
    result = db.collections["recitals"].query(
        query_texts=[user_prompt],
        n_results=3,
    )
    assert len(result["ids"][0]) == 3
    assert isinstance(result["distances"][0][0], float)
    recital_001_index = result["ids"][0].index("recital_001")
    recital_002_index = result["ids"][0].index("recital_002")
    metadata_001 = result["metadatas"][0][recital_001_index]
    assert metadata_001["type"] == "recital"
    metadata_002 = result["metadatas"][0][recital_002_index]
    assert metadata_002["type"] == "recital"


def test_DB_read_base(mock_db, mock_db_cfg):
    db = mock_db.read_mode(config=mock_db_cfg)
    for entity in db.entities:
        assert db.collections[entity].count() == 3


def test_DB_read_invalid_config(mock_db):
    with pytest.raises(AssertionError):
        db = mock_db.read_mode(config=RAGConfig)


def test_DB_build(mock_entity_jsons, mock_db_cfg):
    """
    - mock_entity_jsons must be in arguments (even unused)
    - otherwise the fixture to build the mock jsons is not executed
    """
    db = DB.build_mode(config=mock_db_cfg)
    for entity in db.entities:
        assert db.collections[entity].count() == 3


def test_DB_build_no_files(mock_db_cfg):
    """ do not execute mock json file creation """
    with pytest.raises(FileNotFoundError):
        db = DB.build_mode(config=mock_db_cfg)


def test_DB_build_invalid_config():
    """ do not execute mock json file creation """
    with pytest.raises(AssertionError):
        db = DB.build_mode(config=RAGConfig)


def test_DB_mock_json_creation(mock_entity_jsons, mock_db_cfg):
    """ test json mock data exist at right place for db test """
    save_dir = mock_entity_jsons
    for entity in mock_db_cfg.entities:
        file_path = save_dir / f"{entity}.json"
        assert file_path.exists()
