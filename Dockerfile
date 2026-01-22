# Dockerfile para IBM Cloud Code Engine
# Imagen base PyTorch 2.1.1 precompilada para optimizar build time
FROM pytorch/pytorch:2.1.1-runtime-ubuntu22.04

# Variables de entorno del sistema
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (para Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenjp2-7 \
    libtiff6 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copiar código de la aplicación
COPY *.py ./

# Usuario no-root para ejecución
RUN useradd -m -u 1000 jobuser && chown -R jobuser:jobuser /app
USER jobuser

# Comando de ejecución
ENTRYPOINT ["python", "main.py"]
