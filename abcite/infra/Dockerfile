FROM python:3.11-slim
WORKDIR /opt/abcite
COPY pyproject.toml README.md ./
COPY abcite_core ./abcite_core
COPY agents ./agents
COPY ingest ./ingest
COPY mcp ./mcp
COPY providers ./providers
COPY evals ./evals
COPY apps ./apps
COPY corpus ./corpus
COPY prompts ./prompts
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
