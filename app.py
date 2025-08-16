"""
- when app deployed as docker container on hf spaces, download chromadb from hf datasets
- happens first time when space is accessed by some user after container build
"""
import os
if not os.path.exists("data/chroma_db"):
    print("First run - downloading ChromaDB...")
    os.system("huggingface-cli download kenobijr/eu-ai-act-chromadb --repo-type=dataset --local-dir=data")

import gradio as gr
from src.rag_pipeline import RAGPipeline

# init rag pipeline once at startup
print("... initializing rag pipeline ...")
try:
    rag = RAGPipeline()
    print("... rag pipeline ready ✅")
except Exception as e:
    print(f"Failed to load rag due to: {e}")
    rag = None


def process_query(user_input):
    """
    - process user query through rag pipeline
    - function arguments are bind to gr inputs=.... components
    - function return values are bind to gr outputs=... components
    """
    if not user_input:
        return "Enter a question."

    if rag is None:
        return "Error: RAG pipeline failed to initialize. Please check the logs and restart the application."

    try:
        response = rag.process_query(
            user_prompt=user_input,
            first_query=True,
            rag_enriched=True,
        )
        return response
    except Exception as e:
        return f"Error: {e}"


# gradio interface with custom blocks for enter key support
with gr.Blocks(title="EU AI Act Bot") as demo:
    gr.Markdown("# EU AI Act Bot")
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Ask one question about the EU AI Act and receive one response by the Bot. Reset for new question", lines=3, placeholder="Type in your question here. It may contain a maximum of x chars.")
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Column():
            response_output = gr.Textbox(label="Response", lines=10, interactive=False)
            flag_btn = gr.Button("Flag", variant="secondary")
    
    # submit on button click or enter key
    submit_btn.click(process_query, inputs=user_input, outputs=response_output)
    user_input.submit(process_query, inputs=user_input, outputs=response_output)
    
    # clear functionality
    clear_btn.click(lambda: "", outputs=user_input)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
