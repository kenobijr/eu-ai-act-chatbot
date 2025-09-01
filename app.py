"""
Init RAGPipeline at start -> steers all components:
- RAGPipeline.user_query_len delivers max allowed user query chars based on RAGConfig
- RAGPipeline.process_query takes user query as argument and returns rag-enriched llm response
- input to and output from RAGPipeline are bind to gradio input / response components
- when app deployed as docker container on hf spaces, chromadb is downloaded from hf datasets
- happens first time when space is accessed by some user after container build
"""

import subprocess
import sys
import os
import gradio as gr
from src.rag_pipeline import RAGPipeline

# check if chroma_db was downloaded already -> if not, download it
if not os.path.exists("data/chroma_db"):
    print("First run - downloading ChromaDB...")
    result = subprocess.run(
        ["huggingface-cli", "download", "kenobijr/eu-ai-act-chromadb", 
         "--repo-type=dataset", "--local-dir=data"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to download ChromaDB: {result.stderr}")
        print("The app cannot function without the database.")
        if result.stdout:
            print(f"Output: {result.stdout}")
        sys.exit(1)


# init rag pipeline
print("... initializing rag pipeline ...")
try:
    rag = RAGPipeline()
    print("... rag pipeline ready ✅")
except Exception as e:
    print(f"Failed to load rag due to: {e}")
    rag = None

# get max allowed user query chars from rag pipeline; fallback if rag fails
max_chars = rag.user_query_len if rag else 1000

def process_query(user_input):
    """
    - function arguments are bind to gr inputs=.... components
    - function return values are bind to gr outputs=... components
    """
    if not user_input:
        return "Enter a question."
    if rag is None:
        return "Error: RAG pipeline failed to initialize. Please check logs and restart the app."
    try:
        response = rag.process_query(
            user_prompt=user_input,
            rag_enriched=True,
        )
        return response
    except Exception as e:
        return f"Error: {e}"


# custom theme
theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui"],
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
)

# custom CSS
svg_url = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'>"
    "<text x='50' y='60' font-size='80' text-anchor='middle' fill='rgba(255,127,0,0.05)'>⚖️</text>"
    "</svg>"
)
css = """
body {{
    background-color: #1a1a1a;
    color: #f0f0f0;
    font-family: 'Inter', system-ui;
    background-image: linear-gradient(rgba(255,255,255,0.05), rgba(255,255,255,0.05)),
                      url('{svg_url}');
    background-repeat: repeat;
    background-size: 200px 200px;
}}
.gradio-container {{
    max-width: 1200px;
    margin: auto;
    padding: 2rem 1rem;
    background-color: transparent;
}}
#title {{ text-align: center; color: #ff7f00; font-size: 2.5rem; margin-bottom: 1rem; }}
#instructions {{
    font-size: 0.95rem;
    color: #a0a0a0;
    text-align: center; margin-bottom: 2rem; line-height: 1.5;
}}
.response {{
    background-color: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 1.5rem;
    min-height: 100px;
    height: auto !important;
    overflow: visible !important;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin-top: 1rem;
}}
.input-textbox textarea {{
    background-color: #2a2a2a;
    border: 1px solid #404040;
    color: #f0f0f0;
    resize: vertical;
}}
.button-row {{
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
}}
footer {{ display: none !important; }}  /* Hide Gradio footer for clean look */
"""
css = css.format(svg_url=svg_url)

with gr.Blocks(theme=theme, css=css, title="EU AI Act Bot") as demo:
    gr.Markdown("<h1 id='title'>EU AI Act Bot ⚖️</h1>")

    gr.Markdown("""
    <div id='instructions'>
    Ask precise questions about the EU AI Act and receive RAG-enhanced responses by LLama 3 8B.<br>
    • One question -> One response. Use 'New Question' to start fresh.<br>
    • For best results include references like "Article 17" or "Annex 3" into your prompt.<br>
    </div>""")

    user_input = gr.Textbox(
        label="User Prompt",
        lines=3,
        max_lines=6,
        placeholder=(
            f"Type your question here (e.g., 'Explain obligations under Article 17'). "
            f"Max {max_chars} characters."
        ),
        elem_classes="input-textbox",
        interactive=True,  # ensure typable
        submit_btn=True,  # enable enter to submit
    )

    with gr.Row(elem_classes="button-row"):
        new_question_btn = gr.Button("New Question", variant="secondary", size="lg")
        submit_btn = gr.Button("Submit", variant="primary", size="lg")

    response_output = gr.Textbox(
        label="LLM Response",
        value=None,
        lines=5,
        max_lines=None,  # unlimited to expand fully
        show_copy_button=True,
        interactive=False,
        elem_classes="response",
        placeholder="AI response will appear here in full...",
    )

    # bind events
    submit_btn.click(process_query, inputs=user_input, outputs=response_output)
    user_input.submit(process_query, inputs=user_input, outputs=response_output)
    new_question_btn.click(lambda: ("", ""), outputs=[user_input, response_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
