import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import tiktoken
from src.vector_db import DB
from typing import List, Tuple
from src.config import RAGConfig

# load env variables from .env file
load_dotenv()

# get huggingface token
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")


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
        """
        - getter to deliver token budget for rag context
        - preserves ratio between rag & llm response share
        """
        total_ratio = self.config.llm_response_share + self.config.rag_content_share
        rag_proportion = self.config.rag_content_share / total_ratio
        return int(self.remaining_tokens * rag_proportion)

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


class RAGEngine:
    """
    - core rag logic engine init at RAGPipeline init with reference to same cfg + tm
    - fired up by execute method after user query was processed in RAGPipeline
    - only class in module talking to vectorDB
    """
    def __init__(self, config: RAGConfig, tm: TokenManager):
        self.config = config
        self.tm = tm
        # connect to chromadb in read only mode; if not yet build, run db script before
        self.db = DB(read_only=True)
        self.entities = self.db.entities
        self.remaining_rag_tokens = None
        # collect id's of entities added to rag context to prevent duplicates
        self.used_ids = set()
        self.rag_context = []

    def execute(self, user_prompt: str) -> str:
        # calc token budget for rag context at runtime
        self.remaining_rag_tokens = self.tm.rag_context_tokens
        self._generate_rag_context(user_prompt)
        # format before returning
        return self._format_rag_context(self.rag_context)

    def _generate_rag_context(self, user_prompt: str) -> List[Tuple[str, float]]:
        """
        - compares user prompt against vector_db entries and selects best matches
        - collects best matches until rag token limit is reached or no further good matches
        - return list of tuples with text content & cos distance for each collection entry
        core rag context generation logic:
        - get top nearest entries user_prompt across all entities
        - always grab top 3 nearest articles as "central hub" + their relationships
        - if top 3 articles contain references to other entitites, boost them
        - fill up top relations (minus the 3 already added articles) over all types beginning
          from best distance rating until NOT distance < 0.8 anymore OR rag token limit reached
        """
        # query db for nearest entities (different amounts depending on entity type: -> config)
        nearest_entries = {
            entity: self.db.collection[entity].query(
                query_texts=[user_prompt],
                n_results=getattr(self.config, f"nearest_{entity}"),
            )
            for entity in self.entities
        }

        # RAG CONTEXT BASE: always add top 3 articles (independend of distance)
        # - id, text & distance are appended at class objs as part of final return payload
        # - references of top 3 articles to other entities are saved locally
        top_articles_metadata = []
        for i in range(min(3, len(nearest_entries["articles"]["ids"][0]))):
            # fetch all needed data
            article_id = nearest_entries["articles"]["ids"][0][i]
            text_content = nearest_entries["articles"]["documents"][0][i]
            distance = nearest_entries["articles"]["distances"][0][i]
            metadata = nearest_entries["articles"]["metadatas"][0][i]
            # safety check with text_content of current article, before appending
            tokens = self.tm.get_token_amount(text_content)
            if tokens > self.remaining_rag_tokens:
                break
            # append / add data to containers & update token budget
            self.rag_context.append((text_content, distance))
            self.remaining_rag_tokens -= self.tm.get_token_amount(text_content)
            self.used_ids.add(article_id)
            top_articles_metadata.append(metadata)

        # convert references to other entitites from json str into combined set across entity types
        related_id_sets = self._get_related_id_sets(top_articles_metadata)

        # check and filter nearest entities for relevance -> populate candidates list
        # - if nearest entities are in top 3 articles references, apply relationship boost!
        # - filter out entities with already used_ids & under relevance threshold
        candidates = []
        for entity in self.entities:
            candidate = nearest_entries[entity]
            for i, item_id in enumerate(candidate["ids"][0]):
                # check if item gets relationship boost
                if item_id in related_id_sets:
                    candidate["distances"][0][i] *= self.config.relationship_boost
                # apply filtering criteria: no duplicates; cosine distance for relevance
                if item_id not in self.used_ids and (
                    candidate["distances"][0][i] < self.config.rel_threshold
                ):
                    candidates.append({
                        "content": candidate["documents"][0][i],
                        "distance": candidate["distances"][0][i]
                    })

        # ADDITIONAL RAG CONTEXT: fill up the availble rag context with the best / nearest entities
        if candidates:
            for candidate in sorted(candidates, key=lambda x: x["distance"]):
                # add candidates until rag context tokens run out
                tokens = self.tm.get_token_amount(candidate["content"])
                if tokens <= self.remaining_rag_tokens:
                    self.rag_context.append((candidate["content"], candidate["distance"]))
                    self.remaining_rag_tokens -= tokens

    def _get_related_id_sets(self, top_articles_metadata: list) -> set:
        """
        - extract relationship ids from articles metadata into combined set across all entity types
        - read out from json str; there are no relationships to definiton entity
        """
        related_ids = set()
        rel_entities = ["articles", "recitals", "annexes"]
        for metadata in top_articles_metadata:
            for entity in rel_entities:
                related_ids.update(json.loads(metadata.get(f"related_{entity}", "[]")))
        return related_ids

    @staticmethod
    def _format_rag_context(rag_raw: List[Tuple[str, float]]) -> str:
        """
        - takes (text_content, distance) tuples fo rag retrieved collection entries as input
        - processes & formats them into final str format to make the llm call with
        - measurement for the quality / relevance of rag content is provided to llm
        - convert cosine distance from float to relevance percentage that LLM understands easier
        """
        formatted_chunks = []
        for text, distance in rag_raw:
            relevance = f"{((1 - distance) * 100):.0f}%"
            formatted_chunks.append(f"[Relevance: {relevance}]\n{text}")
        return "\n\n---\n\n".join(formatted_chunks)

    def _reset_state(self) -> None:
        """ resets all components of RAGEngine to enable further user queries"""
        self.used_ids = set()
        self.rag_context = []


class RAGPipeline:
    """
    - steering RAG process using classes RAGEngine, TokenManager and RAGConfig
    - only class in module talking to App / FE and LLM
    - Base Flow:
        1. init RAG Pipeline with VectorDB, Tokenizer, ...
        2. report user_query_len to App / FE
        3. receive user prompt and trigger RAG context creation with it
        4. execute RAG-enriched LLM call
        5. deliver response to App / FE
    """

    def __init__(self, config=None, tm=None, rag_engine=None):
        # init config with default params / systemmessages saved in dataclass
        self.cfg = config if config is not None else RAGConfig()
        # init tokenmanager with passing the config
        self.tm = tm if tm is not None else TokenManager(self.cfg)
        # init rag engine with references to config & tokenmanager
        self.engine = rag_engine if rag_engine is not None else RAGEngine(self.cfg, self.tm)
        # rag context created by RAGEngine after user query is processed
        self.rag_context = []
        # langchain model will be instanciated after rag context gen to calc llm response max_tokens
        self.model = None

    @property
    def user_query_len(self) -> int:
        """
        - tell the app / fe how many chars for user query derived from token manager
        - must be converted from tokens into chars: (4-5 chars/token, ~1 token/word)
        """
        return int(self.tm.user_query_tokens * 4)

    def _validate_user_prompt(self, user_prompt: str) -> bool:
        """ calc token amount of user input str; if valid len return True"""
        assert self.tm.get_token_amount(user_prompt) <= self.tm.user_query_tokens, "Too long prompt"
        assert isinstance(user_prompt, str), "Invalid user prompt data type."
        return True

    def _init_model(self, token_budget: int) -> None:
        """ init langchain model with token budget """
        model = ChatGroq(
            model=self.cfg.llm,
            max_retries=2,
            max_tokens=token_budget,
            temperature=0.7,
            groq_api_key=groq_api_key,
        )
        self.model = model

    def _query_llm(self, user_prompt: str, rag_enriched: bool = True) -> str:
        """
        - if flag rag_enriched with default value True is false, query llm without rag
        - system_prompt selected from config based on rag_enriched flag
        """
        # validation for missing rag context
        if rag_enriched and not self.rag_context:
            raise ValueError("RAG enriched mode requires rag_context to be set")
        # case 1: without rag
        if not rag_enriched:
            messages_base = [
                ("system", self.cfg.system_message_rag_disabled),
                ("human", user_prompt)
            ]
        # case 2: with rag
        else:
            messages_base = [
                ("system", self.cfg.system_message_rag_enabled),
                ("human", f"""Retrieved EU AI Act content with relevance scores:
                {self.rag_context}
                Question: {user_prompt}""")
            ]
        # return llm message for both cases
        return self.model.invoke(messages_base).content

    def _reset_state(self) -> None:
        """ reset all necessary pipeline components to enable further user querys """
        self.tm.reset_state()
        self.engine._reset_state()
        self.rag_context = []
        self.model = None

    def process_query(self, user_prompt: str, rag_enriched: bool = True):
        """
        - central method to process the user_prompt until generating llm response
        - always reset for fresh state at start to enable multiple user queries in app
        """
        self._reset_state()
        # validate user prompt
        if not self._validate_user_prompt(user_prompt):
            raise ValueError("RAG process aborted: too long user prompt or wrong data type")
        # reduce consumed tokens by user prompt
        self.tm.reduce_remaining_tokens(user_prompt)
        # retrieve raw rag_context save at obj
        self.rag_context = self.engine.execute(user_prompt=user_prompt)
        # reduce consumed tokens by rag context
        self.tm.reduce_remaining_tokens(self.rag_context)
        # init langchain model providing remaining tokens to specify max_tokens llm response
        self._init_model(token_budget=self.tm.remaining_tokens)
        # create llm response prompt
        llm_response = self._query_llm(user_prompt=user_prompt, rag_enriched=rag_enriched)

        # testing print delete for production
        print(f"RAG Content: \n{self.rag_context}")
        return llm_response


def main():
    app = RAGPipeline()
    print(f"MAX USER QUERY CHARS: {app.user_query_len}")
    prompt = "How does the EU AI Act protect copyrights of content creators?"
    print(f"USER PROMPT: \n {prompt}")
    print(f"LLM RESPONSE: \n{app.process_query(user_prompt=prompt, rag_enriched=True)}")


if __name__ == "__main__":
    main()
