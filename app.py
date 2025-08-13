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

    try:
        response = rag.process_query(
            user_prompt=user_input,
            first_query=True,
            rag_enriched=True,
        )
        return response
    except Exception as e:
        return f"Error: {e}"


# gradio interface with fn being the function gradio wraps ui around
demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Question about EU AI Act", lines=3),
    outputs=gr.Textbox(label="Response", lines=10),
    title="EU AI Act Advisor"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
