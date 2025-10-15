# COVID-19 Case Prediction - Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/processed data/gluonts results models

# Expose port for potential future dashboard
EXPOSE 8888

# Default command: run help
CMD ["python", "-c", "print('COVID-19 Forecasting Container Ready!\\n\\nAvailable commands:\\n  python src/data_processing/preprocess.py\\n  python src/models/train_baseline.py\\n  python run_pipeline.py\\n\\nOr run: docker exec -it <container> bash')"]

