# Usa la imagen base de NVIDIA CUDA con Ubuntu
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libboost-python-dev \
    libboost-thread-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala Miniconda y Python 3.10 desde conda-forge
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda install -y -c conda-forge python=3.10 && \
    conda clean -ya

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install pycuda numpy && \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "procesadorFiltros.py"]
