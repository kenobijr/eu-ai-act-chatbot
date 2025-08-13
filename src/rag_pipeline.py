import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import tiktoken
from src.vector_db import DB
from typing import List, Tuple


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")


class RAGPipeline:
    """
    - total context window len llama 3 8B = 8192 tokens
    - BUT: groq tokens per minute (TPM): Limit 6000
    - ~7170 tokens -> 1.33 tokens per word -> 5390 words -> 1x DIN A4 page: ~500 words
    """
    # langchain model
    LLM = "llama3-8b-8192"
    # basic param to so set all downstream ratios / token limits work on
    TOTAL_ABSOLUTE_CONTEXT = 6000
    TOKEN_BUFFER = 0.10  # tiktoken used in app for calc  has ~10% less tokens than llama tokeniser
    # fixed cost
    SYSTEM_PROMPT = 350
    FORMATTING_OVERHEAD = 50
    TOTAL_EFFECTIVE_CONTEXT = int(
        TOTAL_ABSOLUTE_CONTEXT * (1 - TOKEN_BUFFER) - SYSTEM_PROMPT - FORMATTING_OVERHEAD
    )
    # general allocations in relation to total_effective_context
    USER_QUERY_SHARE = 0.2
    LLM_RESPONSE_SHARE = 0.2
    RAG_CONTENT_SHARE = 0.6
    assert USER_QUERY_SHARE + LLM_RESPONSE_SHARE + RAG_CONTENT_SHARE == 1.0, "check percentages!"
    # rag relevance threshold for cosine distance -> take only entried from chromadb smaller than
    RELEVANCE_THRESHOLD = 0.8

    def __init__(self):
        # connect to chromadb in read only mode; if not yet build, run db script before
        self.db = DB(read_only=True)
        # langchain model will be instanciated after rag context gen to calc llm response max_tokens
        self.model = None
        # used internally only to check context window limits; tokenising for llm calls done by langchain
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # dynamically updated if tokens were cosumed in steps of the rag process
        self.remaining_tokens = self.TOTAL_EFFECTIVE_CONTEXT
        # max amount tokens for user query;
        self.max_user_query_tokens = self.TOTAL_EFFECTIVE_CONTEXT * self.USER_QUERY_SHARE
        self.entities = ["annexes", "articles", "definitions", "recitals"]
        # structured rag context
        self.rag_context = []

    def count_tokens(self, text: str) -> int:
        """
        - get amount of tokens using tiktokenizer
        - buffer regarding distinct tokenizer of llama are calced into class globals
        """
        return len(self.tokenizer.encode(text))

    def _validate_user_prompt(self, user_prompt: str) -> bool:
        """ calc token amount of user input str; if valid len return True"""
        return self.count_tokens(user_prompt) <= self.max_user_query_tokens and isinstance(user_prompt, str)

    def _update_remaining_tokens(self, consumed_tokens: int) -> None:
        """Update remaining context after consuming tokens"""
        assert self.remaining_tokens - consumed_tokens > 0, "remaining_tokens must be > 0"
        self.remaining_tokens -= consumed_tokens

    def _calc_max_rag_tokens(self) -> int:
        """
        - helper method to calc max token amount for rag context 
        - after user prompt token consume but before llm response consume
        - preserves ratio between LLM_RESPONSE and RAG_CONTENT: 0.25:0.55 = 5:11
        """
        total_ratio = self.LLM_RESPONSE_SHARE + self.RAG_CONTENT_SHARE  # 0.80
        rag_proportion = self.RAG_CONTENT_SHARE / total_ratio  # 0.55/0.80 = 0.6875
        return int(self.remaining_tokens * rag_proportion)

    def _retrieve_context(self, user_prompt: str) -> List[Tuple[str, float]]:
        """
        - compares user prompt against vector_db entries and selects best matches
        - collects best matches until rag token limit is reached or no further good matches
        - rag token limit is delivered by special method and updated in method
        - global token limit self.remaining_tokens is updated outsided this method afterwards
        - return list of tuples with text content & cos distance for each collection entry
        core rag context generation logic while:
        - 1. get top 15 nearest entries across all entities
        - 2. always grab top 3 nearest articles -> they are the "central hub" with relationships
        - 3. fill up top relations (minus the 3 already added articles) over all types beginning
          frombest distance rating until NOT distance < 0.8 anymore OR rag token limit reached
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
        # set up remaining_rag_tokens variable to track consume within method
        remaining_rag_tokens = self._calc_max_rag_tokens()
        # query all collections once -> key value pairs for each entity query return
        all_results = {entity: self.db.collection[entity].query(query_texts=[user_prompt], n_results=15) for entity in self.entities}
        # phase 1: always grab top 3 articles (central hub)
        context = []
        used_ids = set()
        # takes the min of 3 or however many articles were returned -> security from index errors
        for i in range(min(3, len(all_results["articles"]["ids"][0]))):
            # grab text content of entry, add to context container
            text_content = all_results["articles"]["documents"][0][i]
            # construct tuple with text content + distance and append to context container
            context.append((text_content, all_results["articles"]["distances"][0][i]))
            remaining_rag_tokens -= self.count_tokens(text_content)
            # add id of entry to used_id container to prevent duplicates in phase 2
            used_ids.add(all_results["articles"]["ids"][0][i])
        # phase 2: fill with best remaining candidates
        candidates = []
        for entity in self.entities:
            results = all_results[entity]
            for i, item_id in enumerate(results["ids"][0]):
                # apply filtering criteria: no duplicates; cosine distance < 0.8 for relevance
                if item_id not in used_ids and results["distances"][0][i] < self.RELEVANCE_THRESHOLD:
                    # extract only content & disctance into list of dicts to loop & filter easy
                    candidates.append({
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i]
                    })
        # sort by distance and fill while remaining tokens & candidates available
        for candidate in sorted(candidates, key=lambda x: x["distance"]):
            # loop through sorted candidates and add content as tokens for rag context available
            tokens = self.count_tokens(candidate["content"])
            if tokens <= remaining_rag_tokens:
                context.append((candidate["content"], candidate["distance"]))
                remaining_rag_tokens -= tokens
        return context

    def _distance_to_relevance(self, distance: float) -> str:
        """
        - measurement for the quality / relevance of rag content is provided to llm
        - convert cosine distance from float to relevance percentage that LLM understands easier
        """
        relevance = (1 - distance) * 100
        return f"{relevance:.0f}%"

    def _format_rag_context(self, rag_raw: List[Tuple[str, float]]) -> str:
        """
        - takes (text_content, distance) tuples fo rag retrieved collection entries as input
        - processes & formats them into final str format to make the llm call with
        - cosine distance floats mapped into percent strs by sep method
        """
        formatted_chunks = []
        for text, distance in rag_raw:
            relevance = self._distance_to_relevance(distance)
            formatted_chunks.append(f"[Relevance: {relevance}]\n{text}")
        return "\n\n---\n\n".join(formatted_chunks)

    def _init_model(self) -> None:
        """
        - init certain langchain model with llm response max_tokens derived from remaining_tokens
        - saved at obj; respective attribute defined at instanciating
        """
        # Cap max_tokens to ensure total request stays under Groq's 6000 TPM limit
        # Total request = input tokens + max_tokens, so max_tokens should be conservative
        #safe_max_tokens = min(self.remaining_tokens, 1000)  # Cap at 1000 tokens for response
        model = ChatGroq(
            model=self.LLM,
            max_retries=2,
            max_tokens=self.remaining_tokens,
            temperature=0.7,
            groq_api_key=groq_api_key,
        )
        self.model = model

    def _query_llm(self, user_prompt: str, rag_enriched: bool = True) -> str:
        """
        - if flag rag_enriched with default value True is false, query llm without rag
        - system_prompt works for both with rag and without rag
        """
        system_prompt = """You are an expert on the EU AI Act with access to official Act documentation.
        When provided, the retrieved context comes directly from Articles, Recitals, Annexes, and Definitions
        of the EU AI Act which are stored in some VectorDB. This provided material is the 100% legit original content,
        but it can be more or less related to the User prompt depending on the embedding model and matching algorithm.
        The top 3 nearest Articles afte cosine distance are always contained. Additionally, the next nearest n elements
        across all categories / types are added - if they are over the minimum relevance threshold - until RAG token limit is reached.
        This context, if provided, starts with "Retrieved EU AI Act content with relevance scores: " and can contain:
        - Articles: Core legal binding rules and requirements
        - Annexes: Technical specifications and detailed lists (e.g., high-risk AI systems)
        - Definitions: Official terms from Article 3 of the Act
        - Recitals: Background context and rationale (numbered paragraphs from the preamble)
        Each section has a relevance score:
        - 90-100%: Directly answers your question
        - 70-89%: Contains related important information
        - 50-69%: Provides useful context
        - Below 50%: Background information
        Focus on high-relevance sections first, but consider all provided context for a complete answer.
        If the context doesn't contain sufficient information and / or is unrelevant or 
        was not provided at all, you may supplement with your knowledge but indicate this transparently.
        If your knowledge/context does not contain the answer, say so clearly - never make up information!
        The User prompt starts with "Question: "
        """
        if rag_enriched and not self.rag_context:
            raise ValueError("RAG enriched mode requires rag_context to be set")

        # case 1: without rag
        if not rag_enriched:
            messages_base = [
                ("system", system_prompt),
                ("human", user_prompt)
            ]
        # case 2: with rag
        else:
            messages_base = [
                ("system", system_prompt),
                ("human", f"""Retrieved EU AI Act content with relevance scores:

                {self._format_rag_context(self.rag_context)}

                Question: {user_prompt}""")
            ]

        print(self.rag_context)
        print(f"System prompt token len: {self.count_tokens(messages_base[0][1])}")
        print(f"Formatted RAG context + user prompt token len{self.count_tokens(messages_base[1][1])}")
        # return llm message for both cases
        return self.model.invoke(messages_base).content

    def _reset_for_next_query(self) -> None:
        """ set rag_pipeline back as much as necessry to enable further query """
        self.remaining_tokens = self.TOTAL_EFFECTIVE_CONTEXT
        self.rag_context = []
        self.model = None

    def process_query(self, user_prompt: str, first_query: bool = True):
        """
        - central method to process the user_prompt until generating llm response
        - triggers state reset for further user queries depending on flag first_query
        """
        if not first_query:
            self._reset_for_next_query()
        # validate user prompt
        if not self._validate_user_prompt(user_prompt):
            raise ValueError("RAG process aborted: too long user prompt or wrong data type")
        # update remaining tokens after consuming user prompt tokens
        self._update_remaining_tokens(self.count_tokens(user_prompt))
        # retrieve raw rag_context save at obj
        self.rag_context = self._retrieve_context(user_prompt)
        # update remaining tokens after consuming formatted rag context
        self._update_remaining_tokens(self.count_tokens(self._format_rag_context(self.rag_context)))
        # init langchain model with certain llm response max_tokens
        self._init_model()
        # create the rag enriched llm prompt
        llm_response = self._query_llm(user_prompt=user_prompt, rag_enriched=True)
        print(f"LLM response: {llm_response}")



def main():
    app = RAGPipeline()
    app.process_query(user_prompt="What have EU-member countries have to report to the EU about AI highlevel?", first_query=True)
    time.sleep(2)
    app.process_query(user_prompt="I run an Art AI Startup based in the US. Does the EU AI Act effect me in any way?", first_query=False)


if __name__ == "__main__":
    main()
