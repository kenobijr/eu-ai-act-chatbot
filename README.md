## Architecture

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

### Embedding

**Selected Model: sentence-transformers/all-MiniLM-L6-v2**
Decision factors for this 80MB model over larger alternatives (e.g., 420MB all-mpnet-base-v2):
- 5x faster inference: 14,200 vs 2,800 sentences/sec - crucial for responsive user experience on CPU
- Minimal accuracy trade-off: 58.80 vs 63.30 avg performance (~7% difference) - acceptable for legal text retrieval
- Container efficiency: 80MB vs 420MB significantly reduces cold start times on HuggingFace Spaces free tier
- Resource constraints: Optimized for free hosting with limited compute and memory
- 384-dimensional vectors: Good balance between semantic quality and storage efficiency


