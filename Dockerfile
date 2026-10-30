FROM python:3.11-slim

# libgl1 / libglib2.0-0 : requis par OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU d'abord, sinon ultralytics tire la version CUDA (plusieurs Go)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY .streamlit .streamlit
COPY src src
COPY app app
COPY scripts scripts
COPY tests tests

EXPOSE 8501 8512
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
