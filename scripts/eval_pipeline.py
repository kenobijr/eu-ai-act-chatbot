"""
eval pipeline for rag impact benchmarking:
- executes RAGPipeline using 10 questions from assets/questions.txt
- asks each question with rag_enriched=True and rag_enriched=False
- saves answers as numbered txt files in assets/answers/rag_disabled or rag_enabled
- includes 1.5min delay between requests to respect 6k TPM limit
"""

import time
import os
from src.rag_pipeline import RAGPipeline
from typing import List


class EvalPipeline:
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.questions = []
        self.results = []

    def load_questions(self) -> List[str]:
        """load 10 questions from assets/questions.txt"""
        questions_path = "assets/questions.txt"
        try:
            with open(questions_path, "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f if line.strip()]
            print(f"✓ Loaded {len(questions)} questions")
            return questions
        except Exception as e:
            print(f"Error loading questions: {e}")
            return []

    def save_answer(self, question_num: int, question: str, rag_enriched: bool, answer: str):
        """save answer to correct directory with proper format"""
        # determine directory
        dir_name = "rag_enabled" if rag_enriched else "rag_disabled"
        output_dir = f"assets/answers/{dir_name}"
        # ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        # create filename
        filename = f"{question_num}.txt"
        filepath = os.path.join(output_dir, filename)
        # format content
        content = f"""Question: {question}
RAG Enriched: {rag_enriched}

LLM Response:
{answer}"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Error saving answer {question_num} ({dir_name}): {e}")

    def process_question(self, question_num: int, question: str, rag_enriched: bool) -> str:
        """process single question with error handling"""
        mode = "RAG enabled" if rag_enriched else "RAG disabled"
        try:
            answer = self.rag_pipeline.process_query(user_prompt=question, rag_enriched=rag_enriched)
            return answer
        except Exception as e:
            error_msg = f"Error processing question {question_num} ({mode}): {e}"
            print(error_msg)
            return f"ERROR: {error_msg}"

    def run_evaluation(self):
        """main evaluation loop"""
        print("Starting RAG evaluation pipeline...")
        # load questions
        self.questions = self.load_questions()
        if not self.questions:
            print("No questions loaded. Exiting.")
            return
        total_calls = len(self.questions) * 2  # each question asked twice
        call_count = 0
        for i, question in enumerate(self.questions, 1):
            print(f"\nProcessing question {i}/10...")
            # process with RAG disabled
            call_count += 1
            print(f"  [{call_count}/{total_calls}] RAG disabled")
            answer_disabled = self.process_question(i, question, rag_enriched=False)
            self.save_answer(i, question, False, answer_disabled)
            # delay between requests
            if call_count < total_calls:
                print("  Waiting 1.5 min...")
                time.sleep(90)  # 1.5 minutes = 90 seconds
            # process with RAG enabled
            call_count += 1
            print(f"  [{call_count}/{total_calls}] RAG enabled")
            answer_enabled = self.process_question(i, question, rag_enriched=True)
            self.save_answer(i, question, True, answer_enabled)
            # delay between requests (except for last call)
            if call_count < total_calls:
                print("  Waiting 1.5 min...")
                time.sleep(90)
        print(f"\n✓ Evaluation complete! Processed {len(self.questions)} questions.")
        print("Answers saved in assets/answers/rag_disabled and assets/answers/rag_enabled")


def main():
    eval_pipeline = EvalPipeline()
    eval_pipeline.run_evaluation()


if __name__ == "__main__":
    main()
