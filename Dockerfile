# Use Python 3.10 slim as base
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the src directory
COPY src/requirements.txt /app/requirements.txt

# Step 1: Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Step 2: Install PyTorch CPU-only FIRST.
# This prevents sentence-transformers from pulling in 1.3GB+ of NVIDIA CUDA
# libraries (cudnn, nccl, triton etc.) which are useless in a CPU Docker env.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install the rest of the application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# Note: .dockerignore will prevent copying local venv, caches, data, and secrets
COPY . /app/

# Expose port 8000 (Chainlit default port as launched in main.py)
EXPOSE 8000

# Run the app
CMD ["python", "src/main.py", "--app"]
