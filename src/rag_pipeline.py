import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import tiktoken
from src.vector_db import DB

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

    def __init__(self):
        # connect to chromadb in read only mode; if not yet build, run db script before
        self.db = DB(read_only=True)
        # used internally only to check context window limits; tokenising for llm calls done by langchain
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # dynamically updated if tokens were used in steps of the process
        self.remaining_context = self.TOTAL_EFFECTIVE_CONTEXT
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

    def _update_remaining_context(self, consumed_tokens: int) -> None:
        """Update remaining context after consuming tokens"""
        self.remaining_context -= consumed_tokens

    def _get_dynamic_rag_tokens(self) -> int:
        """ calculate max RAG tokens from remaining context, preserving LLM/RAG ratio """
        # ratio between LLM_RESPONSE and RAG_CONTENT: 0.25:0.55 = 5:11
        total_ratio = self.LLM_RESPONSE_SHARE + self.RAG_CONTENT_SHARE  # 0.80
        rag_proportion = self.RAG_CONTENT_SHARE / total_ratio  # 0.55/0.80 = 0.6875
        return int(self.remaining_context * rag_proportion)

    def _retrieve_context(self, user_prompt: str) -> None:
        """
        core rag context generation logic while dynamically updating remaining_context len:
        - 1. get top 15 nearest entries across all entities
        - 2. always grab top 3 nearest articles -> they are the "central hub" with relationships
        - 3. fill up top relations (minus the 3 already added articles) over all types beginning from
        best distance rating until NOT distance < 0.8 anymore OR rag token limit reached
        -> chromadb return this dict structure:
        {
            'ids': [['doc1', 'doc2', 'doc3', ...]],           # List of lists of strings
            'distances': [[0.1, 0.2, 0.3, ...]],             # List of lists of floats
            'documents': [['document text 1', 'document text 2', ...]],  # List of lists of strings
            'metadatas': [[{'key': 'value'}, {'key': 'value'}, ...]],    # List of lists of dicts
            'embeddings': None  # Unless include_embeddings=True
        }
        """
        # get max rag tokens dynamically based on remaining context
        max_rag_tokens = self._get_dynamic_rag_tokens()
        # query all collections once -> key value pairs for each entity query return
        all_results = {entity: self.db.collection[entity].query(query_texts=[user_prompt], n_results=15) for entity in self.entities}
        # phase 1: always grab top 3 articles (central hub)
        context = []
        used_ids = set()
        # set up local remaining_tokens variable which is updated within the method
        remaining_tokens = max_rag_tokens
        # takes the min of 3 or however many articles were returned -> security from index errors
        for i in range(min(3, len(all_results["articles"]["ids"][0]))):
            article_data = {
                "content": all_results["articles"]["documents"][0][i],
                "id": all_results["articles"]["ids"][0][i]
            }
            context.append(article_data)
            used_ids.add(article_data["id"])
            remaining_tokens -= self.count_tokens(article_data["content"])
        # update remaining context obj after consuming top 3 articles
        self._update_remaining_context(max_rag_tokens - remaining_tokens)
        # phase 2: fill with best remaining candidates
        candidates = []
        for entity in self.entities:
            results = all_results[entity]
            for i, item_id in enumerate(results["ids"][0]):
                # apply filtering criteria: no duplicates; cosine distance < 0.8 for relevance
                if item_id not in used_ids and results["distances"][0][i] < 0.8:
                    candidates.append({
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i]
                    })
        # sort by distance and fill while remaining tokens & candidates available
        for candidate in sorted(candidates, key=lambda x: x["distance"]):
            # loop through sorted candidates and add content as tokens for rag context available
            tokens = self.count_tokens(candidate["content"])
            if tokens <= remaining_tokens:
                context.append({"content": candidate["content"]})
                # update local to method remaining tokens
                remaining_tokens -= tokens
                # update instance remaining tokens
                self._update_remaining_context(tokens)
        return context
    
    # do llm call with rag enriched context

    # reset params to make additional rag / llm call on the same object instance



    def execute_rag(self, user_prompt: str):
        # validate user prompt
        if not self._validate_user_prompt(user_prompt):
            raise ValueError("RAG process aborted: too long user prompt or wrong data type")
        # update remaining context after consuming user prompt tokens
        self._update_remaining_context(self.count_tokens(user_prompt))
        # retrieve raw rag_context
        rag = self._retrieve_context(user_prompt)
        print(rag)
        # format and save to self.rag_context
        self.rag_context = "Context:\n" + "\n\n".join(item["content"] for item in rag)
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