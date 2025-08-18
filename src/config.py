from dataclasses import dataclass
import yaml
from pathlib import Path


def _load_system_messages():
    """
    - load 2 systemmessages from yml in data dir at module start
    - save them as global
    """
    load_path = Path(__file__).parent.parent / "data" / "system_messages.yml"
    with open(load_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# saved systemmessages from _load_system_messages
_MESSAGES = _load_system_messages()


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
    system_prompt_rag_disabled: str = _MESSAGES["system_prompt_rag_disabled"]
    system_prompt_rag_enabled: str = _MESSAGES["system_prompt_rag_enabled"]
