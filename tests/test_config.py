import pytest
import yaml
from src.config import _load_system_messages, RAGConfig


def test_RAGConfig_token_share():
    """
    - happy case with default values adding up to 100%
    - less or more than 100% must trigger assertion error on obj instanciation
    """
    cfg = RAGConfig()
    assert cfg.user_query_share + cfg.rag_content_share + cfg.llm_response_share == 1.0
    with pytest.raises(AssertionError):
        cfg = RAGConfig(user_query_share=0.11)
        cfg = RAGConfig(rag_content_share=0.8)


def test_load_system_messages_base(tmp_path):
    # create mock yaml
    data = {"system_message_rag_disabled": "test1", "system_message_rag_enabled": "test2"}
    # pytest fixture tmp_path enables using / syntax from pathlib even this is not imported!
    yaml_file = tmp_path / "test.yml"
    with open(yaml_file, "w") as f:
        yaml.dump(data, f)
    # test
    result = _load_system_messages(str(yaml_file))
    assert result == data


