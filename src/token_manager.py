from src.config import RAGConfig
import tiktoken

class TokenManager:
    """
    - contains all logic related to the token budged for llm prompts
    - tiktoken cl100k_base used for calculations, saved at obj at init
    - calculate initial budget and report to RAGPipeline / FE
    - update budget after consume; inform about current budget
    - getters for user query and rag tokens; for llm response self.remaining_tokens is returned
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # state of available tokens during RAG process; init via _calc_initial_tokens helper
        self.remaining_tokens = self._calc_initial_tokens()
        self.rag_ops_tokens = None

    def _calc_initial_tokens(self) -> int:
        """
        calc effective initial token budget by:
        1. scale total available tokens down by token buffer ratio (different tokenizers)
        2. subtract fixed token amounts reserved for systemmessage and formatting
        """
        return int(
            self.config.total_available_tokens * (1 - self.config.token_buffer) -
            self.config.system_prompt_tokens - self.config.formatting_overhead_tokens
        )

    @property
    def user_query_tokens(self) -> int:
        """ getter to deliver token budget for user query """
        return int(self.remaining_tokens * self.config.user_query_share)

    @property
    def rag_context_tokens(self) -> int:
        """ deliver rag context token budget -> preserve ratio between rag & llm response share """
        total_ratio = self.config.llm_response_share + self.config.rag_content_share
        rag_proportion = self.config.rag_content_share / total_ratio
        return int(self.remaining_tokens * rag_proportion)

    @property
    def llm_response_tokens(self) -> int:
        """ deliver llm response tokens -> eqals self.remaining tokens in this state """
        return self.remaining_tokens

    def get_token_amount(self, text: str) -> int:
        """
        - helper method receiving text and returning token amoutn for it
        - uses tiktokenizer initialised at obj
        """
        return len(self.tokenizer.encode(text))

    def reduce_remaining_tokens(self, text: str) -> None:
        """
        - calc amount tokens for delivered text and subtract at remaining tokens obj
        - happens after user query and rag context creation
        """
        consumed_tokens = self.get_token_amount(text)
        if self.remaining_tokens - consumed_tokens <= 0:
            raise ValueError(f"To few tokens: need {consumed_tokens}, have {self.remaining_tokens}")
        self.remaining_tokens -= consumed_tokens

    def reset_state(self) -> None:
        """ resets remaining tokens to start state enabling multiple user queries in one session """
        self.remaining_tokens = self._calc_initial_tokens()
        self.rag_ops_tokens = None
