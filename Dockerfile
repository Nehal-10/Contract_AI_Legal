# Contract Risk Intelligence System Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY main.py .
COPY config.yaml .

# Create necessary directories
RUN mkdir -p models data logs outputs

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python", "main.py", "--mode", "serve", "--port", "8000"]