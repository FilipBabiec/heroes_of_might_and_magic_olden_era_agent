# CPU image of the Olden Era assistant. It builds both databases from the committed
# data snapshot and bakes in the embedding and reranker models, so it runs offline.
# The LangGraph pipeline needs Ollama, which runs outside the container (see README).
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf

WORKDIR /app

# Dependencies first, so code changes don't reinstall them
COPY requirements-lock.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-lock.txt

COPY ../.. .
RUN python create_sql_db.py \
 && python create_vector_db.py \
 && python -c "import tools; tools.get_reranker()"

ENV HF_HUB_OFFLINE=1

# Default: the MCP server with the game-data tools, over stdio
CMD ["python", "mcp_server.py"]
