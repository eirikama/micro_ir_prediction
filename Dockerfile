FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    wget \
    curl \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv


COPY docker/sqlite-autoconf-3450200.tar.gz /tmp/
RUN cd /tmp && \
    tar xzf sqlite-autoconf-3450200.tar.gz && \
    cd sqlite-autoconf-3450200 && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/sqlite-autoconf-3450200*

ENV LD_PRELOAD=/usr/local/lib/libsqlite3.so.0


WORKDIR /app

COPY requirements.txt .

RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121

RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match \
    -r requirements.txt

COPY src/physics /app/src/physics
RUN cd /app/src/physics && python setup_mie.py build_ext --inplace

COPY . .


CMD ["python", "main.py"]
