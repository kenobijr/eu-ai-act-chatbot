# choose base prefab image
FROM python:3.11-slim

# install system dependencies for os-features of the prefab image
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# create non-root user (hf requirement)
RUN useradd -m -u 1000 user

# set working dir
WORKDIR /app

# copy and install requirements as root
COPY requirements_runtime.txt .
RUN pip install --no-cache-dir -r requirements_runtime.txt

# pre-download the embedding model to avoid runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    model.save('/tmp/models/all-MiniLM-L6-v2')"

# copy app code from cwd to the workdir; --chown ensures non-root user can access/modify files
COPY --chown=user:user . /app

# create data directory with proper permissions
RUN mkdir -p /app/data && \
    chown -R user:user /app && \
    chmod -R 755 /app

# move pre-downloaded model to user's cache directory
RUN mkdir -p /home/user/.cache/torch/sentence_transformers && \
    cp -r /tmp/models/all-MiniLM-L6-v2 /home/user/.cache/torch/sentence_transformers/ && \
    chown -R user:user /home/user/.cache && \
    rm -rf /tmp/models

# switch to non-root user
USER user

# set environment variables
ENV PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME="/home/user/.cache/torch/sentence_transformers" \
    HF_HOME="/home/user/.cache/huggingface"

# expose port for gradio
EXPOSE 7860

# run app.py with python interpreter if container is run
CMD ["python", "app.py"]


