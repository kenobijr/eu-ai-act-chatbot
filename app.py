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

# define theme
theme = gr.themes.Base()

# init rag pipeline once at startup
print("... initializing rag pipeline ...")
try:
    rag = RAGPipeline()
    print("... rag pipeline ready ✅")
except Exception as e:
    print(f"Failed to load rag due to: {e}")
    rag = None

max_chars = rag.user_query_len if rag else 1000  # Fallback if rag fails


def process_query(user_input):
    """
    - process user query through rag pipeline
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


# custom theme: clean & minimalistic
theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'system-ui'],
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
)

# custom CSS for enhanced styling, full response display, wider layout, and subtle legal theme
css = """
body { 
    background-color: #1a1a1a;
    color: #f0f0f0;
    font-family: 'Inter', system-ui;
    background-image: linear-gradient(rgba(255,255,255,0.05), rgba(255,255,255,0.05)), 
                      url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><text x="50" y="60" font-size="80" text-anchor="middle" fill="rgba(255,127,0,0.05)">⚖️</text></svg>');
    background-repeat: repeat;
    background-size: 200px 200px;
}
.gradio-container {
    max-width: 1200px;
    margin: auto;
    padding: 2rem 1rem;
    background-color: transparent;
}
#title { text-align: center; color: #ff7f00; font-size: 2.5rem; margin-bottom: 1rem; }
#instructions { font-size: 0.95rem; color: #a0a0a0; text-align: center; margin-bottom: 2rem; line-height: 1.5; }
.response {
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
}
.input-textbox textarea {
    background-color: #2a2a2a;
    border: 1px solid #404040;
    color: #f0f0f0;
    resize: vertical;
}
.button-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
}
footer { display: none !important; }  /* Hide Gradio footer for clean look */
"""

with gr.Blocks(theme=theme, css=css, title="EU AI Act Bot") as demo:
    gr.Markdown("<h1 id='title'>EU AI Act Bot ⚖️</h1>")

    gr.Markdown("""
    <div id='instructions'>
    Ask precise questions about the EU AI Act and receive expert, RAG-enhanced responses.<br>
    • One question = one response. Use 'New Question' to start fresh.<br>
    • Input limit: {max_chars} characters (approx. {tokens} tokens available).<br>
    • For best results, reference entities like "Article 17", "Annex III", or "Recital 42".<br>
    Powered by Llama 3 via Groq – fast, accurate, and transparent.
    </div>
    """.format(max_chars=max_chars, tokens=rag.tm.user_query_tokens if rag else 'N/A'))

    # user input - above response
    user_input = gr.Textbox(
        label="User Prompt",
        lines=3,
        max_lines=6,
        placeholder=f"Type your question here (e.g., 'Explain obligations under Article 17'). Max {max_chars} characters.",
        elem_classes="input-textbox",
        interactive=True  # ensure typable
    )

    with gr.Row(elem_classes="button-row"):
        new_question_btn = gr.Button("New Question", variant="secondary", size="lg")
        submit_btn = gr.Button("Submit", variant="primary", size="lg")

    # response output - below input/buttons, starts smaller and expands
    response_output = gr.Textbox(
        label="LLM Response",
        value=None,
        lines=5,     # smaller initial height
        max_lines=None,  # unlimited to expand fully
        show_copy_button=True,
        interactive=False,
        elem_classes="response",
        placeholder="Your response will appear here in full..."
    )

    # bind events
    submit_btn.click(process_query, inputs=user_input, outputs=response_output)
    user_input.submit(process_query, inputs=user_input, outputs=response_output)
    new_question_btn.click(lambda: ("", ""), outputs=[user_input, response_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
