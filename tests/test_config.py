import pytest
from pathlib import Path
import yaml
from src.config import _load_system_messages

def test_load_system_messages_base(tmp_path):
    # create mock yaml
    data = {"system_message_rag_disabled": "test1", "system_message_rag_enabled": "test2"}
    yaml_file = tmp_path / "test.yml"
    with open(yaml_file, "w") as f:
        yaml.dump(data, f)
    # test
    result = _load_system_messages(str(yaml_file))
    assert result == data
