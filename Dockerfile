ARG BASE_IMAGE=rocm/pytorch:latest
ARG USE_ROCM=true
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN if [ "$USE_ROCM" = "true" ]; then \
    pip install --no-cache-dir torch-scatter-rocm torch-sparse-rocm; \
    else \
    pip install --no-cache-dir torch-scatter torch-sparse; \
    fi

COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]