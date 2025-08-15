---
title: EU AI Act Bot
emoji: ⚖️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---

## Architecture Decisions

### Scraping
Following decisions reagrding scraping and setup data-models for the entities:

**Additional "Definition" entity**
- In Article 4 are 68 definitons about roles, models, ... which are crucial to understand the content
- To enable granular retrieval of only relevant definitions (2-3 per query) they are extracted and introduced as distinct entity
- By this sending of 68 definitions is avoided and better mapping into embedding vector space
- Claude Opus 4.1 confirmed that this approach makes sense!

**External References:**
- In at least one of the Annexes text contents are URLs directing to external websites
- To keep the entitites and data-model lean, it was decided to not save the external URLs but only the URL-names to keep the text consistent
- External ULRs are mainly contained in the first Annex - not sure if the overall performance of the app could be improved by saving them

**Relationship Architecture:**
- Articles are the "Central Hub"
- One-way relationships from core entity Articles to Annexes, Recitals and other Articles
- No bi-directional relationships to keep data model lean and avoiding circular dependencies

### Vector DB

**DB Setup**
- **Build Strategy**: Separate build-time from runtime for optimal performance
- **Build Script**: `build_vector_db.py` creates and populates the database once
- **Runtime Access**: Application connects to pre-built database (read-only mode)
- **Performance Impact**: 
  - Build time: ~30-60 seconds (one-time, includes embedding generation)
  - Runtime: <1 second (instant connection to existing database)
- **Storage**: Persisted to `data/chroma_db/` using ChromaDB's native format

**Embedding Model: sentence-transformers/all-MiniLM-L6-v2**
Decision factors for this 80MB model over larger alternatives (e.g., 420MB all-mpnet-base-v2):
- 5x faster inference: 14,200 vs 2,800 sentences/sec - crucial for responsive user experience on CPU
- Minimal accuracy trade-off: 58.80 vs 63.30 avg performance (~7% difference) - acceptable for legal text retrieval
- Container efficiency: 80MB vs 420MB significantly reduces cold start times on HuggingFace Spaces free tier
- Resource constraints: Optimized for free hosting with limited compute and memory
- 384-dimensional vectors: Good balance between semantic quality and storage efficiency

**HNSW Index Configuration: Cosine distance**
- ChromaDB uses cosine distance, which is calculated as cosine_distance = 1 - cosine_similarity
    - Cosine similarity = 1: cosine distance = 0 (perfect match)
    - Cosine similarity = 0.7: cosine distance = 0.3
    - Cosine similarity = 0: cosine distance = 1
    - Cosine similarity = -1: cosine distance = 2

### RAG Pipeline

**tiktoken cl100k_base to check prompt / context length**
- With tiktoken (cl100k_base): ~95% accurate for Llama counting -> 1MB dependency
- With native Llama 3 8B tokenizer: 100% accurate -> transformers + tokenizer 750MB dependencies!
- Tiktokenizer + 15 % buffer

**Conversation flow: 1 question -> 1 response**
- Due to context window size of LLM of ~8,192 tokens, the app supports no follow-up questions
- User asks → RAG retrieves → LLM answers → Done
- Focus on RAG enriched context to enable one clear response -> all other options would drive complexity with context window restraints

**Model**: Llama 3 8B Instruct with ~8,192 token context window

### Deployment Strategy

**Single Repository with Runtime Data Loading**
- Maintain single GitHub repository as source of truth - no sync issues or duplicate code maintenance
- Use .dockerignore to exclude development files (tests/, build scripts, etc.) from Docker image
- ChromaDB files stored separately in HuggingFace Dataset repository to bypass 10MB file size limit
- Runtime download of ChromaDB from HuggingFace Dataset on first startup
- Grok 4 and Claude Opus 4 consensus: cleanest solution for free-tier constraints!

**Docker Optimization with .dockerignore**
- Key exclusions to minimize image size: Documentation, Scripts not needed at runtime, ...
- Exclude VectorDB to avoid having to use Git Large-File Service
- Expected Docker image: ~1-2GB (vs ~500MB+ for ChromaDB files)

**HuggingFace Datasets for Large File Storage for ChromaDB Files**
- Bypasses HuggingFace Spaces 10MB file limit without Git LFS complexity
- Free, persistent storage that survives Space restarts (vs ephemeral 50GB disk)
- Clean separation: application code in Spaces, data in Datasets
- Download cached after first retrieval - minimal performance impact


