FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# install packages first so this slow layer is cached when only the code changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# a normal user, not root
RUN useradd --create-home --uid 1000 app

# code, labels, word list and the trained model
COPY bangla_common.py app.py labels.json lexicon.txt ./
COPY models ./models
RUN chown -R app:app /app

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
