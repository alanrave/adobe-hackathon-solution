FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY persona_analyzer.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Pre-download the sentence transformer model to stay within size limits
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Make the script executable
RUN chmod +x persona_analyzer.py

# Run the analysis script
CMD ["python", "persona_analyzer.py"]