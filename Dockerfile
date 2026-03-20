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
    python3-pip \
    build-essential \
    wget \
    curl \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

# build sqlite from local file — no network dependency
COPY docker/sqlite-autoconf-3450200.tar.gz /tmp/
RUN cd /tmp && \
    tar xzf sqlite-autoconf-3450200.tar.gz && \
    cd sqlite-autoconf-3450200 && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/sqlite-autoconf-3450200*

# force python to use the newly built sqlite
ENV LD_PRELOAD=/usr/local/lib/libsqlite3.so.0

# verify sqlite version — fails build if too old
RUN echo "import sqlite3\nv = sqlite3.sqlite_version_info\nassert v >= (3,31,0), f'too old: {sqlite3.sqlite_version}'\nprint('SQLite:', sqlite3.sqlite_version)" | python


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

RUN cd src/Physics && python setup_mie.py build_ext --inplace

CMD ["python", "main.py"]