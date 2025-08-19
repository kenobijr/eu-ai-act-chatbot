import os
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
    - calculate initial budget and report to RAGPipeline / FE
    - update budget after consume; inform about current budget
    - getters for user query and rag tokens; for llm response self.remaining_tokens is returned
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        # save tokenizer at obj to enable fast calls
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

    def reset_remaining_tokens(self) -> None:
        """ resets remaining tokens to start state enabling multiple user queries in one session """
        self.remaining_tokens = self._calc_initial_tokens()


class RAGEngine:
    """
    - core rag logic receiving user prompt and token budget, returning final rag context as str
    - only this class inits & queries the vectorDB
    """
    def __init__(self, config: RAGConfig, tm: TokenManager):
        #self.user_prompt = user_prompt
        self.config = config
        self.tm = tm
        # connect to chromadb in read only mode; if not yet build, run db script before
        self.db = DB(read_only=True)
        self.entities = ["annexes", "articles", "definitions", "recitals"]
        self.token_budget = None

    def execute(self, user_prompt: str) -> str:
        # calc token budget for rag context at runtime
        self.token_budget = self.tm.rag_context_tokens
        return self._generate_rag_context(user_prompt)

    def _generate_rag_context(self, user_prompt: str) -> List[Tuple[str, float]]:
        """
        - compares user prompt against vector_db entries and selects best matches
        - collects best matches until rag token limit is reached or no further good matches
        - return list of tuples with text content & cos distance for each collection entry
        core rag context generation logic while:
        - 1. get top 15 nearest entries across all entities
        - 2. always grab top 3 nearest articles as "central hub" + their relationships
        - 3. apply relationship boosts to all candidates based on connections to top 3 articles
        - 4. fill up top relations (minus the 3 already added articles) over all types beginning
          from best distance rating until NOT distance < 0.8 anymore OR rag token limit reached
        -------------------------------------------------------------
        -> from chromadb you get this dict structure for collection queries:
        {
            'ids': [['doc1', 'doc2', 'doc3', ...]],           # List of lists of strings
            'distances': [[0.1, 0.2, 0.3, ...]],             # List of lists of floats
            'documents': [['document text 1', 'document text 2', ...]],  # List of lists of strings
            'metadatas': [[{'key': 'value'}, {'key': 'value'}, ...]],    # List of lists of dicts
            'embeddings': None  # Unless include_embeddings=True
        }
        """
        # query all collections once -> key value pairs for each entity query return
        all_results = {
            entity:
            self.db.collection[entity].query(
                query_texts=[user_prompt],
                n_results=15,
                #include=["documents", "distances", "metadatas", "ids"],  # include metadata
            )
            for entity in self.entities
        }
        # phase 1: always grab top 3 articles (central hub) + collect their relationships
        context = []
        used_ids = set()
        # takes the min of 3 or however many articles were returned -> security from index errors
        for i in range(min(3, len(all_results["articles"]["ids"][0]))):
            # grab text
            text_content = all_results["articles"]["documents"][0][i]
            # meta = all_results["articles"]["metadatas"][0][i]
            # construct tuple with text content + distance and append to context container
            context.append((
                text_content,
                all_results["articles"]["distances"][0][i],
                # all_results["articles"]
            ))
            # safety clause? or even before appending?
            self.token_budget -= self.tm.get_token_amount(text_content)
            # add id of entry to used_id container to prevent duplicates in phase 2
            used_ids.add(all_results["articles"]["ids"][0][i])

        # phase 2: fill with best remaining candidates across all entities
        # step 1:
        # - filter for used id's of step 1 & relevance threshold
        # - extract only certain data from each entry: content & distance
        candidates = []
        for entity in self.entities:
            results = all_results[entity]
            for i, item_id in enumerate(results["ids"][0]):
                # apply filtering criteria: no duplicates; cosine distance < 0.8 for relevance
                if item_id not in used_ids and results["distances"][0][i] < self.config.rel_threshold:
                    # extract only content & disctance into list of dicts to loop & filter easy
                    candidates.append({
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i]
                    })
        # step 2: sort by distance and fill while remaining tokens & candidates available
        for candidate in sorted(candidates, key=lambda x: x["distance"]):
            # loop through sorted candidates and add content as tokens for rag context available
            tokens = self.tm.get_token_amount(candidate["content"])
            if tokens <= self.token_budget:
                context.append((candidate["content"], candidate["distance"]))
                self.token_budget -= tokens

        return self._format_rag_context(context)

    @staticmethod
    def _format_rag_context(rag_raw: List[Tuple[str, float]]) -> str:
        """
        - takes (text_content, distance) tuples fo rag retrieved collection entries as input
        - processes & formats them into final str format to make the llm call with
        - cosine distance floats mapped into percent strs by sep method
        """
        formatted_chunks = []
        for text, distance in rag_raw:
            relevance = RAGEngine._distance_to_relevance(distance)
            formatted_chunks.append(f"[Relevance: {relevance}]\n{text}")
        return "\n\n---\n\n".join(formatted_chunks)

    @staticmethod
    def _distance_to_relevance(distance: float) -> str:
        """
        - measurement for the quality / relevance of rag content is provided to llm
        - convert cosine distance from float to relevance percentage that LLM understands easier
        """
        relevance = (1 - distance) * 100
        return f"{relevance:.0f}%"


class RAGPipeline:
    """
    - total context window len llama 3 8B = 8192 tokens
    - ~7170 tokens -> 1.33 tokens per word -> 5390 words -> 1x DIN A4 page: ~500 words
    - BUT: groq tokens per minute (TPM): Limit 6000
    """

    def __init__(self):
        # init config with default params / systemmessages saved in dataclass
        self.config = RAGConfig()
        # init tokenmanager with passing the config
        self.tm = TokenManager(self.config)
        # init rag engine with references to config & tokenmanager
        self.rag_engine = RAGEngine(self.config, self.tm)
        # rag context created by RAGEngine after user query is processed
        self.rag_context = []
        # langchain model will be instanciated after rag context gen to calc llm response max_tokens
        self.model = None


    def user_query_len(self) -> int:
        """
        - tell the app / fe how many chars for user query derived from token manager
        - must be converted from tokens into chars: (4-5 chars/token, ~1 token/word)
        """
        return int(self.tm.user_query_tokens * 4)

    def _validate_user_prompt(self, user_prompt: str) -> bool:
        """ calc token amount of user input str; if valid len return True"""
        assert self.tm.get_token_amount(user_prompt) <= self.tm.user_query_tokens, "Too long user prompt."
        assert isinstance(user_prompt, str), "Invalid user prompt data type."
        return True

    def _init_model(self, token_budget: int) -> None:
        """ init langchain model with token budget """
        model = ChatGroq(
            model=self.config.llm,
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
                ("system", self.config.system_message_rag_disabled),
                ("human", user_prompt)
            ]
        # case 2: with rag
        else:
            messages_base = [
                ("system", self.config.system_message_rag_enabled),
                ("human", f"""Retrieved EU AI Act content with relevance scores:
                {self.rag_context}
                Question: {user_prompt}""")
            ]
        # return llm message for both cases
        return self.model.invoke(messages_base).content

    def _reset_for_fresh_state(self) -> None:
        """ set rag_pipeline back as much as necessry to enable further query """
        self.tm.reset_remaining_tokens()
        self.rag_context = []
        self.model = None

    def process_query(self, user_prompt: str, rag_enriched: bool = True):
        """
        - central method to process the user_prompt until generating llm response
        - always reset for fresh state at start to enable multiple user queries in app
        """
        self._reset_for_fresh_state()
        # validate user prompt
        if not self._validate_user_prompt(user_prompt):
            raise ValueError("RAG process aborted: too long user prompt or wrong data type")
        # reduce consumed tokens by user prompt
        self.tm.reduce_remaining_tokens(user_prompt)
        # retrieve raw rag_context save at obj
        self.rag_context = self.rag_engine.execute(user_prompt=user_prompt)
        # reduce consumed tokens by rag context
        self.tm.reduce_remaining_tokens(self.rag_context)
        # init langchain model providing remaining tokens to specify max_tokens llm response
        self._init_model(token_budget=self.tm.remaining_tokens)
        # create the rag enriched llm prompt
        llm_response = self._query_llm(user_prompt=user_prompt, rag_enriched=rag_enriched)
        
        # testing print delete for production
        print(f"RAG_FORMATTED: \n{self.rag_context}")
        print(f"LLM response: \n{llm_response}")
        return llm_response


def main():
    app = RAGPipeline()
    prompt = "My AI startup incorporated in France tracks by facial recognition if certain fabric workers are underperformers, do I face any regulation?"
    app.process_query(user_prompt=prompt, rag_enriched=True)


if __name__ == "__main__":
    main()
