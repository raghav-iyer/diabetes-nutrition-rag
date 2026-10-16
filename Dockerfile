# App image: the nrag CLI and (later) the API / Streamlit app.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[pdf]"

COPY config.yaml ./
COPY data/guidelines/registry.yaml ./data/guidelines/registry.yaml

# Data, caches and secrets are mounted at runtime, never baked into the image.
RUN useradd --create-home --uid 10001 nrag && chown -R nrag /app
USER nrag
ENTRYPOINT ["nrag"]
CMD ["--help"]
