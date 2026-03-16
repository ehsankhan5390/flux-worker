FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY flux_worker.py .

CMD ["python","flux_worker.py"]
