# Use the official Python 3.11 lightweight base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /code

# Copy requirements and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user (Required for Hugging Face Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy all project code into the container
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose port 7860 (Default port for Hugging Face Spaces)
EXPOSE 7860

# Command to launch the FastAPI server on port 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]