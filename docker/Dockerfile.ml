FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for ML
COPY ml/train/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all ML code
COPY ml/ ./ml/

# Train models during build (ensures models exist)
RUN cd /app/ml/train && python3 simple_train.py

# Set Python path
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/ml/train/models
ENV PORT=8001

EXPOSE 8001

# Run the ML prediction service
CMD ["python3", "-m", "ml.serve.predictor"]
