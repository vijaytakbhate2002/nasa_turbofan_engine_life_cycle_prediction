FROM python:3.11-slim  

WORKDIR /mlops_project

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . /mlops_project

CMD ["python", "src\\data_operation.py"]

CMD ["python", "data_processing_pipeline.py"]

CMD ["python", "model_training_pipeline.py"]