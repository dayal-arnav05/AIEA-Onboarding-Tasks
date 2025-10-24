# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY task_3/cifar10train.py /workspace/cifar10train.py

# Create directory for data and models
RUN mkdir -p /workspace/data /workspace/models

# Set the default command
CMD ["python", "cifar10train.py"]

