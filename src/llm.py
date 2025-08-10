import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")


def main():
    llm = ChatGroq(
        model="llama3-8b-8192",
        max_retries=2,
        max_tokens=None,
        temperature=0.7,
        groq_api_key=groq_api_key,
    )

    system_prompt = """You are an expert on the EU AI Act. Answer questions based solely
    on your knowledge and context if provided. If your knowledge/context do not contain
    the answer, say so clearly."""

    messages_base = [
        ("system", system_prompt),
        ("human", "Who are you and what's your quest?")
    ]
    answer_base = llm.invoke(messages_base).content

    print(answer_base)


if __name__ == "__main__":
    main()
