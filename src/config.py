from dataclasses import dataclass
import yaml
from pathlib import Path


def _load_system_messages(file_path: str):
    """
    - load 2 systemmessages from yml in data dir at module start
    - save them as global
    """
    load_path = Path(__file__).parent.parent / file_path
    try:
        with open(load_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"could not load system_messages.yml at: {load_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"invalid yml format for system_messages.yml: {e}") from e
    except PermissionError as e:
        raise PermissionError(f"permission error with system_messages.yml at: {load_path}") from e
    except Exception as e:
        raise RuntimeError(f"error loading system_messages.yml: {type(e).__name__}: {e}") from e


# saved systemmessages from _load_system_messages
_MESSAGES = _load_system_messages(file_path="data/system_messages.yml")


@dataclass
class RAGConfig:
    """
    - class to steer all relevant RAGPipeline parameters central
    - provides also systemmessages for cases non-rag & rag-enriched
    """
    # langchain model
    llm: str = "llama3-8b-8192"
    # context window and token management
    total_absolute_context: int = 6000
    token_buffer: float = 0.10
    system_prompt_tokens: int = 550
    formatting_overhead: int = 50
    # allocation ratios
    user_query_share: float = 0.2
    llm_response_share: float = 0.2
    rag_content_share: float = 0.6
    # rag relevance threshold for cosine distance
    rel_threshold: float = 0.8
    # relationship boost factors (multiplication)
    related_article_boost: float = 0.75
    related_recital_boost: float = 0.85
    related_annex_boost: float = 0.80
    # system prompts
    system_message_rag_disabled: str = _MESSAGES["system_message_rag_disabled"]
    system_message_rag_enabled: str = _MESSAGES["system_message_rag_enabled"]

    def __post_init__(self):
        # make sure token share percentages add up to 100%
        assert (
            self.user_query_share
            + self.llm_response_share
            + self.rag_content_share
            == 1.0
        ), "token share percentages percentages don't add up to 100%, check config."
