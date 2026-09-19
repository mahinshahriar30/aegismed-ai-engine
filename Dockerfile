# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/code

WORKDIR ${WORKDIR}

# Install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create persistent storage directory for ChromaDB
RUN mkdir -p ${WORKDIR}/chroma_db

# Create and switch to non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser ${WORKDIR}
USER appuser

# Expose Render default port
EXPOSE 10000

# Launch FastAPI backend via Uvicorn using Render's $PORT
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}"]