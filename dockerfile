FROM python:3.11-slim   # Use 3.11 (widely supported in ML ecosystem)

WORKDIR /mlops_project

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /mlops_project

EXPOSE 5000

CMD ["python", "app.py"]
