import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import tiktoken
from src.vector_db import DB
from typing import List

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")


class RAGPipeline:
    """
    - 
    """
    # total context window
    TOTAL_ABSOLUTE_CONTEXT = 8192  # max token amount by llama 3 8B
    TOKEN_BUFFER = 0.10  # tiktoken used in app for calc  has ~10% less tokens than llama tokeniser
    # fixed cost
    SYSTEM_PROMPT = 100  # "You are an expert on EU AI Act..."
    FORMATTING_OVERHEAD = 100  # "Context:\n", "Question:", separators
    # total effective space for user-query, llm-response & rag-context & allocation on init
    # ~7170 tokens -> 1.33 tokens per word -> 5390 words -> 1x DIN A4 page: ~500 words
    TOTAL_EFFECTIVE_CONTEXT = int(
        TOTAL_ABSOLUTE_CONTEXT * (1 - TOKEN_BUFFER) - SYSTEM_PROMPT - FORMATTING_OVERHEAD
    )
    # general allocations in relation to total_effective_context
    USER_QUERY_SHARE = 0.2
    LLM_RESPONSE_SHARE = 0.25
    RAG_CONTENT_SHARE = 0.55
    # rag relevance threshold for cosine distance -> take only entried from chromadb smaller than
    RELEVANCE_THRESHOLD = 0.8

    def __init__(self):
        # connect to chromadb in read only mode; if not yet build, run db script before
        self.db = DB(read_only=True)
        # used internally only to check context window limits; tokenising for llm calls done by langchain
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # dynamically updated if tokens were cosumed in steps of the rag process
        self.remaining_tokens = self.TOTAL_EFFECTIVE_CONTEXT
        # max amount tokens for user query;
        self.max_user_query_tokens = self.TOTAL_EFFECTIVE_CONTEXT * self.USER_QUERY_SHARE
        self.entities = ["annexes", "articles", "definitions", "recitals"]
        # rag context to be added to the llm call
        self.rag_context = ""

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
        self.remaining_tokens -= consumed_tokens

    def _calc_max_rag_tokens(self) -> int:
        """ calculate max RAG tokens from remaining context, preserving LLM/RAG ratio """
        # ratio between LLM_RESPONSE and RAG_CONTENT: 0.25:0.55 = 5:11
        total_ratio = self.LLM_RESPONSE_SHARE + self.RAG_CONTENT_SHARE  # 0.80
        rag_proportion = self.RAG_CONTENT_SHARE / total_ratio  # 0.55/0.80 = 0.6875
        return int(self.remaining_tokens * rag_proportion)

    def _retrieve_context(self, user_prompt: str) -> List[str]:
        """
        - compares user prompt against vector_db entries and selects best matches
        - collects best matches until rag token limit is reached or no further good matches
        - rag token limit is delivered by special method and updated dynamically
        - return list of raw text content from collection entries; formatting in sep method
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
            # grab text content of entry, add to context container & update remaining tokens
            text_content = all_results["articles"]["documents"][0][i]
            context.append(text_content)
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
                context.append(candidate["content"])
                # update local to method remaining tokens
                remaining_rag_tokens -= tokens
                # update instance remaining tokens
                self._update_remaining_tokens(tokens)
        return context

    def _format_rag_context(self, rag_raw: List[str]) -> str:
        """
        - takes raw rag context as list of str
        - processes & formats it for llm call
        """
        return "Context:\n" + "\n\n".join(text for text in rag_raw)


    
    # do llm call with rag enriched context

    # reset params to make additional rag / llm call on the same object instance



    def execute_rag(self, user_prompt: str):
        # validate user prompt
        if not self._validate_user_prompt(user_prompt):
            raise ValueError("RAG process aborted: too long user prompt or wrong data type")
        # update remaining tokens after consuming user prompt tokens
        self._update_remaining_tokens(self.count_tokens(user_prompt))
        # retrieve raw rag_context, format it by method & save at obj
        self.rag_context = self._format_rag_context(self._retrieve_context(user_prompt))
        # update remaining tokens after consuming formatted rag context tokens
        self._update_remaining_tokens(self.count_tokens(self.rag_context))
        
        
        print(f"RAG context prepared: {self.count_tokens(self.rag_context)} tokens")
        print(self.rag_context)



def main():
    app = RAGPipeline()
    app.execute_rag("How are high-risk ai systems defined in EU AI Act context?")
    print(app.rag_context)


if __name__ == "__main__":
    main()

    # llm = ChatGroq(
    #     model="llama3-8b-8192",
    #     max_retries=2,
    #     max_tokens=None,
    #     temperature=0.7,
    #     groq_api_key=groq_api_key,
    # )

    # system_prompt = """You are an expert on the EU AI Act. Answer questions based solely
    # on your knowledge and context if provided. If your knowledge/context do not contain
    # the answer, say so clearly."""

    # messages_base = [
    #     ("system", system_prompt),
    #     ("human", "Who are you and what's your quest?")
    # ]
    # answer_base = llm.invoke(messages_base).content

    # print(answer_base)

    #     db_entities = ["annexes", "articles", "definitions", "recitals"]
    # db = DB(db_entities)
    
    # -----------testcases
    # test definitions query -> "definition_03"
    # results_def = db.collection["definitions"].query(
    #     query_texts=["provider"],
    #     n_results=1,
    # )
    # print(results_def)
    # # test recitals query -> "recital_010"
    # results_rec = db.collection["recitals"].query(
    #     query_texts=["The fundamental right to the protection of personal data is safeguarded in particular by Regulations (EU) 2016/679[11] and (EU) 2018/1725[12]"],
    #     n_results=1,
    # )
    # print(results_rec)
    # # test annex query -> "annex_03"
    # results_annex = db.collection["annexes"].query(
    #     query_texts=["High-risk AI systems pursuant toArticle 6(2) are the AI systems listed in any"],
    #     n_results=1,
    # )
    # print(results_annex)
    # # test article query -> "annex_002"
    # results_article = db.collection["articles"].query(
    #     query_texts=["1. This Regulation applies to:(a) providers placing on the market or putting"],
    #     n_results=1,
    # )
    # print(results_article)