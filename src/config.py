from dataclasses import dataclass, field
import yaml
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime


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
    """ token management, sytemmessages, direct & semantic search parameters """
    # langchain model
    llm: str = "llama3-8b-8192"
    # system prompts
    system_message_rag_disabled: str = _MESSAGES["system_message_rag_disabled"]
    system_message_rag_enabled: str = _MESSAGES["system_message_rag_enabled"]
    # context window and token management
    total_available_tokens: int = 6000  # groq TPM limit = 6k
    token_buffer: float = 0.12  # reduce token budget by ratio: tiktoken / llama 3 tokenizer
    system_prompt_tokens: int = 550
    formatting_overhead_tokens: int = 50
    # token allocation ratios derived from total available token amount
    user_query_share: float = 0.2
    llm_response_share: float = 0.2
    rag_content_share: float = 0.6
    # direct search -> max matches within one user prompt; search patterns / entities
    max_direct_matches: int = 3  # maximum number of direct entity matches to extract from prompt
    search_configs: dict = field(default_factory=lambda: {
        "article": {
            "collection": "articles", "max_num": 113, "pad_digits": 3, "id_prefix": "article_"
        },
        "recital": {
            "collection": "recitals", "max_num": 180, "pad_digits": 3, "id_prefix": "recital_"
        },
        "annex": {
            "collection": "annexes", "max_num": 13, "pad_digits": 2, "id_prefix": "annex_"
        },
    })
    # semantic search -> get x nearest matches from db per entity type; filter down by relevance
    nearest_articles: int = 30  # total number 113
    nearest_annexes: int = 5  # total number 13
    nearest_recitals: int = 15  # total number 180
    nearest_definitions: int = 15  # total number 68
    rel_threshold: float = 0.8  # rag relevance threshold for cosine distance
    # relationship boost -> multiplicate cosine distance of boosted entities by factor
    relationship_boost: float = 0.5

    def __post_init__(self):
        """ make sure token share percentages add up to 100% """
        assert (
            self.user_query_share
            + self.llm_response_share
            + self.rag_content_share
            == 1.0
        ), "token share percentages percentages don't add up to 100%, check config."


@dataclass
class DBConfig:
    """
    - embedding functions parameters work on:
        - from chromadb.utils import embedding_functions
        - embedding_functions.SentenceTransformerEmbeddingFunction
    """
    # normal list not allowed in data class due to mutability
    entities: list[str] = field(
        default_factory=lambda: ["annexes", "articles", "definitions", "recitals"]
    )
    # embedding
    embedding_function = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device = "cpu"
    # unit to measure distance between entity in high dim space
    measurement_unit = "cosine"
    # raw data loading - relative to project root
    data_dir: Path = Path(__file__).parent.parent / "data" / "raw"
    file_extension: str = ".json"
    # data saving
    save_dir: Path = Path(__file__).parent.parent / "data" / "chroma_db"


@dataclass
class SCRAPEConfig:
    """
    - all parameters related to scraping - most important entity counts
    - rate limit calls and rate limit period must be set at _make_requests decorator directly
    - css selectors in the code at the specific entities
    """
    base_url: str = "https://artificialintelligenceact.eu"
    # CRUCIAL -> entity count ranges -> check on website!!
    recital_range_start: int = 0
    recital_range_end: int = 180
    annex_range_start: int = 0
    annex_range_end: int = 13
    article_range_start: int = 0
    article_range_end: int = 113
    # special case: article to scrape out definitions
    definitions_article_id: int = 3
    # rate limiting and request parameters
    request_timeout: int = 20
    random_delay_min: float = 0.01
    random_delay_max: float = 0.2
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    # data / output
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "raw"
    )
    file_extension: str = ".json"


@dataclass
class Article:
    """
    - core legal binding rules together with annexes
    - central hub for relationships (one-way from articles to other entities)

    """
    id: str
    title: str
    text_content: str
    chapter_title: str
    section_title: Optional[str] = None
    entry_date: Optional[str] = None
    related_recital_ids: List[str] = field(default_factory=list)
    related_annex_ids: List[str] = field(default_factory=list)
    related_article_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not all([self.id, self.title, self.text_content, self.chapter_title]):
            raise ValueError(f"Article {self.id} missing required fields")


@dataclass
class Annex:
    """
    - detailed technical information, lists, classifications
    - crucial for practitioners (e.g., high-risk AI use cases)
    """
    id: str
    title: str
    text_content: str

    def __post_init__(self):
        if not all([self.id, self.title, self.text_content]):
            raise ValueError(f"Annex {self.id} missing required fields")


@dataclass
class Recital:
    """
    - part of preamble/introduction, no legal binding directives
    - articles with directives are derived from these
    """
    id: str
    text_content: str

    def __post_init__(self):
        if not self.id or not self.text_content:
            raise ValueError(f"Recital {self.id} requires id and text_content")


@dataclass
class Definition:
    """
    - extract 68 definitions from Article 3 as separate entity
    - enable granular retrieval of relevant definitions
    """
    id: str
    title: str
    text_content: str

    def __post_init__(self):
        if not all([self.id, self.title, self.text_content]):
            raise ValueError(f"Definition {self.id} missing required fields")


@dataclass
class ScrapeMeta:
    """ scrape meta .json is created at scraping process """
    scraped_date: datetime
    source_url: str
    entity_counts: Dict[str, int] = field(default_factory=dict)
