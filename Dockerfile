# Use the official Python 3.11 lightweight base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /code

# Copy requirements and install dependencies without caching wheels
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Create a non-root user (Required for Hugging Face Spaces & security best practices)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user app home
WORKDIR $HOME/app

# Copy all project code into the container with correct ownership
COPY --chown=user . $HOME/app

# Ensure writable directory for ChromaDB storage
RUN mkdir -p $HOME/app/chroma_db

# Expose default backend port
EXPOSE 10000

# Launch Uvicorn server, using dynamic PORT env var if available
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}"]