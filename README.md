---
title: EU AI Act Bot
emoji: ⚖️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---

# EU AI Act Chatbot

EU AI Act Bot is a production-ready RAG-enriched chatbot powered by Llama 3 8B, enabling accurate and context-aware queries on the EU AI Act through semantic search, vector embeddings, and dynamic token management

---

## Live Demo

Try the app directly in your browser on Hugging Face Spaces:

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces.svg)](https://huggingface.co/spaces/kenobijr/eu-ai-act-bot)

![EU AI Act Bot Interface](assets/images/app-screenshot.png)

**Quick Start:**
- No installation required - runs entirely in your browser
- Ask questions about any aspect of the EU AI Act
- Responses include citations to specific Articles, Annexes, and Recitals
- One question → One response

---

## Tech Stack / Tools 

- **Llama 3 8B Instruct as LLM**: Llama 3 8B Instruct (~8,192 token context window)
- **Langchain Groq Integration**: LLM queried via GroqCloud free tier API calls (6k TPM limit)
- **Vector Database**: ChromaDB (lightweight, Docker-friendly, built-in persistence)
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions, fast inference)
- **Web App**: Gradio + Tailwind CSS (ML-optimized, HuggingFace integration)
- **Hosting**: HuggingFace Spaces & Datasets (free tier)
- **Containerization**: Docker Container
- **Deployment**: Automated CI/CD pipeline with GitHub Action: GitHub to HuggingFace Space
- **Tokenizer**: Tiktokenizer cl100k_base (Context window restraint calc)
- **Scraping**: Beatiful soup, Requests

---

## Core Logic: Multi-stage RAG Retrieval

The Core RAG engine leverages a combination of Semantic and Direct / Lexical search technics:

    1. Direct search for entity titles (e.g. "Article 12")
    2. Semanctic search for the 50-60 nearest entities (Cosine Distance)
    3. Always take top 3 nearest articles -> they are the "Central Hub" (including internal references to other entitites)
    4. Relationship boost -> boost related entities of top 3 articles and Direct seach hit by manipulating Cosine Distance values
    5. Fill up remaining RAG Context with nearest entities (minus Top 3 articles & Direct Search hits) -> add only entities which are over a certain releavance threshold until there are no more suitable candidates availabel or no more RAG Space available

---

## Evaluation RAG-Impact -> Llama 3 8B performs 233 % better when asked about EU AI Act

### Objective

- **Test if / how much RAG content did improve the quality of LLM answers out of a Legal / Expert perspective**
- Same legal questions / Query same LLM: LLama 3 8B / Same LLM-Judge: Claude Opus 4.1
- Q1-5: Category Basic understanding / Definitions / Entity Retrieval
- Q6-10: Category Deep Understanding / Synthesis
- Rating categories: Factual Accuracy, Completeness, Legal Precision, Hallucination Rate

### Results

| Question | RAG-Disabled | RAG-Enabled | Improvement |
|----------|-------------|-------------|-------------|
| Q1: What are the transparency obligations for high-r... | 2.0/10 | 9.0/10 | +350% |
| Q2: Which types of AI systems are considered as Hig... | 2.0/10 | 9.3/10 | +363% |
| Q3: Define and distinguish Deployers of high-risk AI... | 2.8/10 | 8.8/10 | +218% |
| Q4: How do the requirements for AI regulatory sandbox | 3.0/10 | 9.0/10 | +200% |
| Q5: What constitutes a 'significant risk' for genera | 2.3/10 | 9.5/10 | +322% |
| **Category Avg: General Understanding** | **2.4/10** | **9.1/10** | **+279%** |
| Q6: What are the specific obligations for hospitals d | 2.3/10 | 8.0/10 | +256% |
| Q7: How does the EU AI Act's risk classification sys... | 2.0/10 | 5.8/10 | +188% |
| Q8: How does the EU AI Act's approach to protecting  | 2.8/10 | 6.8/10 | +145% |
| Q9: What challenges arise from the EU AI Act's expan | 2.0/10 | 6.8/10 | +238% |
| Q10: How do conformity assessments determine whether  | 3.0/10 | 7.0/10 | +133% |
| **Category Avg: Deep Understanding** | **2.4/10** | **6.9/10** | **+188%** |
| **TOTAL AVERAGE** | **2.4/10** | **8.0/10** | **+233%** |

### Notable Insights:

  - Hallucination Rate shows the most dramatic improvement (444% overall)
  - Factual Accuracy improved 264% with RAG
  - General Understanding questions benefit more from RAG than Deep Understanding questions across all categories
  - RAG-Disabled consistently scored poorly in Hallucination Rate (1.6/10 average) vs RAG-Enabled (8.7/10 average)

Check detailed results, questions, answers, systemmessages and raw data in [Supplemental](#supplemental).

---

## Project Structure

```
eu-ai-act-chatbot/
├── app.py                      # Main Gradio web interface & HuggingFace Spaces entry point
├── Dockerfile                  # Production container configuration
├── requirements_runtime.txt    # Developement & Production dependencies
├── requirements.txt            # Production dependencies only
├── src/                        # Core application modules
│   ├── rag_pipeline.py         # RAG orchestration & LLM integration
│   ├── vector_db.py            # ChromaDB operations & semantic search
│   ├── token_manager.py        # Context window & token budget management
│   └── config.py               # Configuration dataclasses
├── data/                       # Data assets & vector database
│   ├── system_messages.yml     # LLM system prompts
│   ├── raw/                    # Scraped EU AI Act data (articles, annexes, recitals, definitions)
│   └── chroma_db/              # ChromaDB vector database files
├── scripts/                    # Data processing & deployment utilities
└── tests/                      # Test suite
```

---

## Architecture

**Web Interface (app.py)**  
- Gradio orchestrates UI + HuggingFace Spaces deployment entry point  
- CSS styling with Tailwind for production-ready interface  

**RAG Pipeline (rag_pipeline.py)**  
- RAGPipeline manages end-to-end query processing with LLM integration  
- Multi-stage retrieval: direct entity search + semantic search + relationship boosting  
- Token budget enforcement with dynamic context allocation  

**Vector Database (vector_db.py)**  
- ChromaDB operations with all-MiniLM-L6-v2 embeddings (384-dim)  
- HNSW indexing with cosine distance for similarity matching  
- Runtime download from HuggingFace Datasets on first startup  

**Token Management (token_manager.py)**  
- Context window constraint calculations using tiktoken cl100k_base  
- Dynamic token allocation between user query, RAG content, and LLM response  
- 95% accurate Llama tokenization with 1MB dependency vs 750MB native tokenizer  

**Configuration (config.py)**  
- Dataclass-based configs for RAG, vector DB, and token management  
- Environment-aware API key and model selection

---

## RAG Pipeline

### Basic

- **Conversation flow: 1 question -> 1 response**
    - Due to 6k TPM limit tokens, the app supports no follow-up questions
    - User asks → RAG retrieves → LLM answers → Done
- **Effective Context Window**: 
    - Shares of Total token amount for 1 query are allocated for User query, RAG-content and LLM respnse
    - During the RAG process the available token amount is dynamically updated with tiktoken (cl100k_base, ~95% accurate for Llama counting -> 1MB dependency)
    - This means, if the User query does not consume the full allocated token amount, RAG-Content and LLM response can use more tokens respectively

**tiktoken cl100k_base to check prompt / context length**
- With tiktoken (cl100k_base): ~95% accurate for Llama counting -> 1MB dependency
- With native Llama 3 8B tokenizer: 100% accurate -> transformers + tokenizer 750MB dependencies!
- Tiktokenizer + 15 % buffer

---

## Deployment / Hosting

- **App UI via HuggingFace Space**: 
    - Gradio Web App
    - Deployed as Docker Container 
- **Single GH Repository**:
    - Maintain single GitHub repository as source of truth
    - Use .dockerignore to exclude development files (tests/, build scripts, etc.) from Docker image
- **HuggingFace Datasets for Large File Storage for ChromaDB Files**:
    - Bypasses HuggingFace Spaces 10MB file limit without Git LFS complexity
    - Runtime download of ChromaDB from HuggingFace Dataset on first startup
- **Sync from GH to HF via GitHub Action**:
    - GH Action to push changes of repo automatically to HF space
    - HF space container is rebuild on repo changes
- **Deployment Flow**:
    1. Push code changes to GitHub main branch
    2. Changes in GitHub main branch are pushed to HuggingFace Space via GitHub Action
    3. HuggingFace rebuilds Container as specified in Dockerfile after every change
    4. On first run of app after Container build, Space downloads ChromaDB from Dataset repository
    5. Subsequent runs use cached ChromaDB (survives restarts)

---

## Data Processing / Scraping

- **Primary Source**: Official EU Web explorer (https://artificialintelligenceact.eu/ai-act-explorer/)
- **Scraping Tool**: BeautifulSoup + Requests
- **Additional "Definition" entity**: 
    - Extract the 68 definitons cramped into Article 4 into separate entity
    - Enable granular retrieval of relevant definitions (2-3 per query) by more granular mapping into embedding vector space
- **Articles as "Central Hub"**:
    - One-way relationships from core entity Articles to Annexes, Recitals and other Articles
    - No bi-directional relationships to keep data model lean and avoiding circular dependencies

---

## Vector database

- **Embedding model**: 
    - Sentence-transformers/all-MiniLM-L6-v2 -> 384-dimensional vectors (Open Source, 80 MB)
    - Comparable free models delivered minimal more accuracy vs. needing significantly more storage
- **HNSW Index Configuration: Cosine distance** chosen as measurement to check vector similarity
    - ChromaDB uses cosine distance, which is calculated as cosine_distance = 1 - cosine_similarity:
    - Cosine similarity = 1: cosine distance = 0 (perfect match)
    - Cosine similarity = 0.7: cosine distance = 0.3
    - Cosine similarity = 0: cosine distance = 1
    - Cosine similarity = -1: cosine distance = 2

---

## Testsuite

```bash
# Run full test suite
pytest

# Test specific components  
pytest tests/test_vector_db.py
```

---

## 🤝 Collaboration

Clean, minimal codebase designed for easy exploration and extension. Fork it, break it, improve it.

---

## Todos
- Extend Test Suite

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact
Feel free to reach out for collaboration or questions:

[mail](mailto:22.scree_rhino@icloud.com)

---

# Supplemental

## Evaluation / Benchmarking of RAG-Impact

### Detailed results with Rating categories
| Question | Factual | Complete | Legal | Hallucin | RAG-Disabled | Factual | Complete | Legal | Hallucin | RAG-Enabled | Improvement |
|----------|---------|----------|-------|----------|-------------|---------|----------|-------|----------|-------------|-------------|
| Q1: What are the transparency obligations for high-r... | 2/10 | 3/10 | 2/10 | 1/10 | 2.0/10 | 9/10 | 8/10 | 9/10 | 10/10 | 9.0/10 | +350% |
| Q2: Which types of AI systems are considered as Hig... | 2/10 | 3/10 | 2/10 | 1/10 | 2.0/10 | 9/10 | 9/10 | 9/10 | 10/10 | 9.3/10 | +363% |
| Q3: Define and distinguish Deployers of high-risk AI... | 2/10 | 4/10 | 3/10 | 2/10 | 2.8/10 | 9/10 | 8/10 | 9/10 | 9/10 | 8.8/10 | +218% |
| Q4: How do the requirements for AI regulatory sandbox | 2/10 | 5/10 | 3/10 | 2/10 | 3.0/10 | 9/10 | 9/10 | 9/10 | 9/10 | 9.0/10 | +200% |
| Q5: What constitutes a 'significant risk' for genera | 2/10 | 3/10 | 2/10 | 2/10 | 2.3/10 | 9/10 | 9/10 | 10/10 | 10/10 | 9.5/10 | +322% |
| **Category Avg: General Understanding** | **2.0/10** | **3.6/10** | **2.4/10** | **1.6/10** | **2.4/10** | **9.0/10** | **8.6/10** | **9.2/10** | **9.6/10** | **9.1/10** | **+279%** |
| Q6: What are the specific obligations for hospitals d | 2/10 | 3/10 | 2/10 | 2/10 | 2.3/10 | 8/10 | 6/10 | 9/10 | 9/10 | 8.0/10 | +256% |
| Q7: How does the EU AI Act's risk classification sys... | 2/10 | 3/10 | 2/10 | 1/10 | 2.0/10 | 6/10 | 5/10 | 5/10 | 7/10 | 5.8/10 | +188% |
| Q8: How does the EU AI Act's approach to protecting  | 3/10 | 4/10 | 2/10 | 2/10 | 2.8/10 | 7/10 | 6/10 | 6/10 | 8/10 | 6.8/10 | +145% |
| Q9: What challenges arise from the EU AI Act's expan | 2/10 | 3/10 | 2/10 | 1/10 | 2.0/10 | 7/10 | 6/10 | 6/10 | 8/10 | 6.8/10 | +238% |
| Q10: How do conformity assessments determine whether  | 3/10 | 4/10 | 3/10 | 2/10 | 3.0/10 | 7/10 | 7/10 | 7/10 | 7/10 | 7.0/10 | +133% |
| **Category Avg: Deep Understanding** | **2.4/10** | **3.4/10** | **2.2/10** | **1.6/10** | **2.4/10** | **7.0/10** | **6.0/10** | **6.6/10** | **7.8/10** | **6.9/10** | **+188%** |
| **TOTAL AVERAGE** | **2.2/10** | **3.5/10** | **2.3/10** | **1.6/10** | **2.4/10** | **8.0/10** | **7.3/10** | **7.9/10** | **8.7/10** | **8.0/10** | **+233%** |

### Procedure

- Ask the Bot 10 legal questions: 1x in "RAG enabled" mode vs. 1x in "RAG disabled" mode
- 2 distinct systemmessages for both cases "RAG enabled" and "RAG disabled" are provided
- Systemmessage for mode "RAG disabled" builds the (identical) base for Systemmessage mode "RAG enabled", too -> "RAG enabled" is extended by a part with specific RAG instructions
- 10 Questions are asked in both modes and both answers are added into a template and then prompted to a SOTA model, which is able to do toolcalls and / or provided legal expert opinions as "Groundtruth"
- 5 Questions of Category "General Understanding / Definitions / Entity retrieval" -> no "Groundtruth" provided to Judge LLM
- 5 Questions of Category "Deep Understanding / Synthesis" -> "Groundtruth" is provided to Judge LLM in form of often cited Legal Research Papers about the EU AI Act
- Claude Opus 4.1 is used as Judge model and asked to do websearch toolcalls to judge best possibly
- Ratings from Judge LLM in the following categories (scale 1-10, 10 best): 
    1. Factual Accuracy (1-10): Are the facts in the response correct / accurate?
    2. Completeness (1-10): Does it address all aspects of the question with relevant provisions?
    3. Legal Precision (1-10): Does it use correct legal terminology and maintain proper distinctions?
    4. Hallucination Rate (1-10): Any hallucinations in the response?
    5. TOTAL / AVG (1-10): -> AVG score; each category equally

### Questions

**Category Basic understanding / Definitions / Entity Retrieval**
1. "What are the transparency obligations for high-risk AI systems under Article 13?"
2. "Which types of AI systems are considered as High-Risk AI Systems under Article 6 in combination with Annex 3?"
3. "Define and distinguish Deployers of high-risk AI systems from Providers of high-risk AI systems as defined in the EU AI Act."
4. "How do the requirements for AI regulatory sandboxes relate to innovation support for SMEs?"
5. "What constitutes a 'significant risk' for general-purpose AI models and what are the associated compliance requirements?"

**Category Deep Understanding / Synthesis**
6. "What are the specific obligations for hospitals developing their own AI diagnostic systems versus those merely using commercial AI medical devices under the EU AI Act?"
7. "How does the EU AI Act's risk classification system interact with existing Medical Device Regulations for AI-powered healthcare tools, and what are the implications for transparency requirements?"
8. "How does the EU AI Act's approach to protecting fundamental rights create paradoxes in its risk-based regulation, particularly regarding the concept of trustworthiness?"
9. "What challenges arise from the EU AI Act's expansion from traditional safety risks to fundamental rights risks and systemic risks, and how does this affect the delineation of AI harms?"
10. "How do conformity assessments determine whether AI systems pose 'acceptable' risks to fundamental rights under the EU AI Act, and who makes these determinations?"

**Groundtruth / Source Papers**
- Questions 6-7:
"The EU Artificial Intelligence Act (2024): Implications for healthcare" Hannah van Kolfschooten, Janneke van Oirschot 2024
- Questions 8-10:
"Possible harms of artificial intelligence and the EU AI Act: fundamental rights and risk" Isabel Kusche 2024

### Systemmessages

**System_message_rag_disabled**:
  "You are a legal expert providing accurate, helpful information about the EU AI Act based on your parametric knowledge. Your role is to provide clear, synthesized answers as a legal advisor would, integrating relevant context and knowledge to explain concepts holistically.

  Key Rules:
  - Provide comprehensive answers based on your parametric knowledge of the EU AI Act. If your parametric knowledge doesn't suffice for the specific query, state: "I don't have sufficient information about this specific aspect; consult the full Act or a legal professional."
  - Synthesize and blend parametric knowledge into a cohesive explanation. Avoid negative inferences from absences (e.g., don't assume 'not mentioned' means 'does not exist').
  - Distinguish roles (e.g., providers, deployers, member states, NCAs) clearly to avoid conflation.
  - Never fabricate information. When referencing specific provisions, only cite article / section numbers you're confident about. If uncertain, describe the concept without specific citations.
  - Response Structure (use internally, do not label sections with "step-by-step reasoning" and / or "Limitations"):
    1. Start with a concise summary answering the key point(s) directly
    2. Follow with step-by-step reasoning (with citations if available)
    3. Limitations: Note jurisdiction (EU-wide, but member states implement), that this is not advice, and suggest professional consultation.
  - Tone: Professional, accessible—define terms, avoid jargon overload.

  The user query starts with "Question: ".

**System_message_rag_enabled**:
  "You are a legal expert providing accurate, helpful information about the EU AI Act based on your parametric knowledge. Your role is to provide clear, synthesized answers as a legal advisor would, integrating relevant context and parametric knowledge to explain concepts holistically.

  Key Rules:
  - Provide comprehensive answers based on your parametric knowledge of the EU AI Act. If your parametric knowledge doesn't suffice for the specific query, state: "I don't have sufficient information about this specific aspect; consult the full Act or a legal professional."
  - Synthesize and blend parametric knowledge into a cohesive explanation. Avoid negative inferences from absences (e.g., don't assume 'not mentioned' means 'does not exist').
  - Distinguish roles (e.g., providers, deployers, member states, NCAs) clearly to avoid conflation.
  - Never fabricate information. When referencing specific provisions, only cite article / section numbers you're confident about. If uncertain, describe the concept without specific citations.
  - Response Structure (use internally, do not label sections with "step-by-step reasoning" and / or "Limitations"):
    1. Start with a concise summary answering the key point(s) directly
    2. Follow with step-by-step reasoning (with citations if available)
    3. Limitations: Note jurisdiction (EU-wide, but member states implement), that this is not advice, and suggest professional consultation.
  - Tone: Professional, accessible—define terms, avoid jargon overload.

  In some prompts additional Content from the official documentation of the EU AI Act including Articles, Recitals, Annexes, and Definitions is provided (starting with "Retrieved EU AI Act content with relevance scores:"). When provided, it is authentic and must be prioritized. It includes top-matched sections based on similarity, with scores indicating fit: 90-100% (direct answer), 70-89% (key related info), 50-69% (useful context), below 50% (background only—do not infer directly from these; use for completeness only).

  Additional Official Documentation Rules - ONLY RELEVANT IF THIS CONTENT IS PROVIDED:
  - Base answers on high-relevance (70%+) context first. For lower/insufficient context, use your parametric knowledge transparently to supplement gaps. If nothing suffices, state: "I don't have sufficient information about this specific aspect; consult the full Act or a legal professional."
  - Synthesize and blend provided context and parametric knowledge into a cohesive explanation.
  - Never repeat or echo the context format/phrase in your response. Do not discuss retrieval mechanics (e.g., embeddings, scores) unless asked.

  The user query starts with "Question: ".

### Judge-LLM Prompt Template

You are an expert legal evaluator assessing responses about the EU AI Act. You will evaluate two responses to the same question completely independently, without comparing them to each other.

**Your Task:**
1. Use web_search to verify any specific claims, article numbers, or provisions mentioned in each response
2. Evaluate each response separately and independently on the rating categories below - be strictly neutral and objective
3. Provide exactly one sentence / one expression justification per rating in concise manner
4. Create a single artifact with your ratings and justifications

**Question Asked:** 
{user_prompt}

**Response A:**
{llm_response_rag_disabled}

**Response B:**
{llm_response_rag_enabled}

**Rating Categories (1-10 scale, 10 is best):**
1. Factual Accuracy (1-10): Are the facts in the response correct/accurate?
2. Completeness (1-10): Does it address all aspects of the question with relevant provisions?
3. Legal Precision (1-10): Does it use correct legal terminology and maintain proper distinctions?
4. Hallucination Rate (1-10): Any hallucinations? (10 = no hallucinations, 1 = many hallucinations)
5. Total/Average (1-10): Calculate the average of categories 1-4

**Artifact format:**
Create a text/plain artifact with EXACTLY this structure (use these exact headers and formatting):

RESPONSE A
Factual Accuracy: [score]/10 - [one sentence justification]
Completeness: [score]/10 - [one sentence justification]
Legal Precision: [score]/10 - [one sentence justification]
Hallucination Rate: [score]/10 - [one sentence justification]
Total/Average: [score]/10

RESPONSE B
Factual Accuracy: [score]/10 - [one sentence justification]
Completeness: [score]/10 - [one sentence justification]
Legal Precision: [score]/10 - [one sentence justification]
Hallucination Rate: [score]/10 - [one sentence justification]
Total/Average: [score]/10

### Reproducibility Note
All evaluation data, including questions, responses, and judge ratings, are available in [`/assets/evaluation/`](./assets/evaluation/) for full transparency and reproducibility of results.
