# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for WeasyPrint and other packages
RUN apt-get update && apt-get install -y \
    libcairo2 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Spacy model
RUN python -m spacy download en_core_web_md

# Copy the rest of the application
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start FastAPI backend in the background\n\
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &\n\
\n\
# Wait a few seconds for backend to start\n\
sleep 5\n\
\n\
# Start Streamlit frontend\n\
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
