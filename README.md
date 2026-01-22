# IBM Image Recognition Job

Job para generar embeddings de imágenes en IBM Cloud Code Engine.

## Descripción

Este Job procesa imágenes almacenadas en IBM Cloud Object Storage (COS), genera embeddings usando ResNet50 preentrenado, agrupa los embeddings por producto y guarda los resultados en COS.

### Workflow

1. **Conectar a COS**: Establece conexión segura a IBM Cloud Object Storage
2. **Listar imágenes**: Escanea el bucket y agrupa imágenes por producto (estructura: `producto-<sku>/imagen.*`)
3. **Procesar imágenes**:
   - Validar formato (.webp, .jpg, .jpeg, .png)
   - Convertir a RGB si es necesario
   - Redimensionar a 224x224 (estándar ImageNet)
   - Aplicar normalización
4. **Generar embeddings**: Usar ResNet50 (2048 dimensiones)
5. **Agregar embeddings**: Promediar embeddings por producto
6. **Guardar resultados**: Guardar en COS en formato .npy y .json
7. **Idempotencia**: No reprocesar productos que ya tienen embeddings

## Estructura del Proyecto

```
.
├── main.py              # Orquestador principal
├── cos_client.py        # Cliente IBM COS
├── image_utils.py       # Validación y carga de imágenes
├── embedding_model.py   # Generación de embeddings
├── requirements.txt     # Dependencias Python
├── Dockerfile           # Imagen para Code Engine
└── README.md            # Este archivo
```

## Requisitos

- Python 3.11+
- IBM Cloud COS (Credentials)
- Acceso a IBM Cloud Code Engine
- GPU (opcional, recomendado para velocidad)

## Variables de Entorno Requeridas

```env
# IBM COS Configuration
COS_ENDPOINT=https://s3.us-south.cloud-object-storage.appdomain.cloud
COS_ACCESS_KEY_ID=xxx...
COS_SECRET_ACCESS_KEY=xxx...
COS_BUCKET_NAME=mi-bucket

# Configuración opcional
COS_OUTPUT_PREFIX=embeddings/          # Prefijo para guardar embeddings
MODEL_NAME=resnet50                     # resnet50, resnet101, resnet152
```

## Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/marceloluvianconsiss/ibm-image-recognition-job.git
cd ibm-image-recognition-job

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
export COS_ENDPOINT="..."
export COS_ACCESS_KEY_ID="..."
export COS_SECRET_ACCESS_KEY="..."
export COS_BUCKET_NAME="..."

# Ejecutar Job
python main.py
```

## Ejecución en IBM Cloud Code Engine

### 1. Preparar imagen Docker

```bash
# Construir imagen
docker build -t ibm-image-recognition-job:v1.0.0 .

# Etiquetar para registry (ICR)
docker tag ibm-image-recognition-job:v1.0.0 \
  <region>.icr.io/<namespace>/ibm-image-recognition-job:v1.0.0

# Logear en ICR
ibmcloud cr login

# Push a registry
docker push <region>.icr.io/<namespace>/ibm-image-recognition-job:v1.0.0
```

### 2. Crear Job en Code Engine

```bash
# Login a IBM Cloud
ibmcloud login
ibmcloud target -g <resource-group>

# Crear Job
ibmcloud ce job create \
  --name image-recognition-job \
  --image <region>.icr.io/<namespace>/ibm-image-recognition-job:v1.0.0 \
  --env COS_ENDPOINT=<endpoint> \
  --env COS_ACCESS_KEY_ID=<key> \
  --env COS_SECRET_ACCESS_KEY=<secret> \
  --env COS_BUCKET_NAME=<bucket> \
  --env COS_OUTPUT_PREFIX=embeddings/ \
  --env MODEL_NAME=resnet50 \
  --memory 4G \
  --cpu 2
```

### 3. Ejecutar Job

```bash
# Ejecutar una sola vez
ibmcloud ce jobrun submit --job image-recognition-job

# Monitorear ejecución
ibmcloud ce jobrun logs --jobrun <jobrun-name>

# Ver status
ibmcloud ce jobrun get --jobrun <jobrun-name>
```

### 4. (Opcional) Programar ejecuciones

```bash
# Crear EventArc trigger (ej: diario a las 2 AM)
ibmcloud ce eventarc create \
  --name daily-image-recognition \
  --destination-type job \
  --destination image-recognition-job \
  --event-schedule "0 2 * * *"
```

## Estructura de Datos en COS

### Input
```
bucket/
├── producto-SKU001/
│   ├── img_01.webp
│   ├── img_02.jpg
│   └── img_03.png
├── producto-SKU002/
│   ├── image_a.jpg
│   └── image_b.jpeg
└── ...
```

### Output
```
bucket/
├── embeddings/
│   ├── producto-SKU001/
│   │   ├── embeddings.npy        # numpy array (2048 dimensiones)
│   │   └── embeddings.json       # JSON con metadata
│   ├── producto-SKU002/
│   │   ├── embeddings.npy
│   │   └── embeddings.json
│   └── ...
```

### Formato de Salida

**embeddings.npy**:
- Array de numpy serializado
- Formato: float32
- Tamaño: (2048,) para ResNet50

**embeddings.json**:
```json
{
  "product_sku": "producto-SKU001",
  "embeddings": [0.123, 0.456, ...],
  "shape": [2048],
  "dtype": "float32"
}
```

## Logs y Monitoreo

El Job genera logs detallados en stdout:

```
2025-01-22 10:30:00 - __main__ - INFO - ============================================================
2025-01-22 10:30:00 - __main__ - INFO - Iniciando Job: IBM Image Recognition
2025-01-22 10:30:00 - __main__ - INFO - ============================================================
2025-01-22 10:30:01 - cos_client - INFO - Conectado a COS: mi-bucket
2025-01-22 10:30:02 - embedding_model - INFO - Cargando modelo: resnet50
2025-01-22 10:30:05 - __main__ - INFO - Encontrados 5 productos
2025-01-22 10:30:15 - __main__ - INFO - ✓ Producto completado: producto-SKU001
...
2025-01-22 10:31:00 - __main__ - INFO - Job completado:
2025-01-22 10:31:00 - __main__ - INFO -   - Productos procesados: 5
2025-01-22 10:31:00 - __main__ - INFO -   - Errores: 0
2025-01-22 10:31:00 - __main__ - INFO - ============================================================
```

### Ver logs en Code Engine

```bash
# Logs en tiempo real
ibmcloud ce jobrun logs --jobrun <jobrun-name> -f

# Últimos 100 líneas
ibmcloud ce jobrun logs --jobrun <jobrun-name> --tail 100
```

## Idempotencia

El Job es **idempotente**:
- Si se ejecuta dos veces, no reprocesa productos
- Verifica existencia de `embeddings.npy` antes de procesar
- Si ya existen embeddings, omite el producto
- Permite re-ejecución segura sin corromper datos

```python
# Verificación en main.py
if cos_client.check_embeddings_exist(product_sku, format="npy"):
    logger.info(f"Embeddings ya existen para {product_sku}, omitiendo...")
    continue
```

## Manejo de Errores

El Job maneja:
- ✓ Conexión a COS fallida → Falla inmediata
- ✓ Imágenes inválidas → Omite y continúa
- ✓ Formato no soportado → Omite y continúa
- ✓ Error en una imagen → Omite producto actual
- ✓ Sin imágenes en bucket → Finaliza sin error

Exit codes:
- `0`: Éxito (todos los productos procesados)
- `1`: Error crítico o algunos productos fallaron

## Optimizaciones en Producción

### Memoria y CPU

Para grandes volúmenes:
```bash
ibmcloud ce job create \
  --memory 8G \      # Aumentar memoria para batch processing
  --cpu 4 \          # Usar múltiples CPUs
  ...
```

### GPU (si disponible en su región)

```bash
ibmcloud ce job create \
  --gpu 1 \          # Incluir GPU para faster inference
  --memory 16G \
  ...
```

### Batch size (modificar embedding_model.py)

```python
# Procesar imágenes en lotes para menor memoria
batch_size = 32
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    embeddings = model.get_batch_embeddings(batch)
```

## Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| `ValueError: Faltan variables de entorno` | Variables COS no configuradas | Verificar `--env` en `job create` |
| `ConnectionError: Unable to connect to COS` | Endpoint incorrecto o credenciales inválidas | Validar endpoint y credenciales en IAM |
| `Out of memory` | Bucket muy grande o GPU saturada | Aumentar `--memory`, usar CPU mode |
| `PIL.UnidentifiedImageError` | Imagen corrupta en bucket | Validar formato de imagen en COS |
| `No se encontraron imágenes` | Estructura de carpetas incorrecta | Verificar formato: `producto-SKU/imagen.ext` |

## Desarrollo

### Ejecutar en local

```bash
# Con variables de entorno
python main.py

# Con dotenv
pip install python-dotenv
# Crear .env local (no commitar)
python main.py
```

### Testing (opcional)

```bash
# Validar conexión COS
python -c "from cos_client import COSClient; c = COSClient(); print(c.list_product_images())"

# Validar modelo
python -c "from embedding_model import get_embedding_model; m = get_embedding_model(use_gpu=False)"
```

## Límites Conocidos

- **Tamaño máximo de imagen**: 100MB (límite de COS SDK)
- **Productos simultáneos**: 1 por ejecución (procesamiento secuencial)
- **Modelos soportados**: ResNet50, ResNet101, ResNet152
- **Formatos de imagen**: .webp, .jpg, .jpeg, .png

## Contacto y Soporte

Para issues o sugerencias: [GitHub Issues](https://github.com/marceloluvianconsiss/ibm-image-recognition-job/issues)

---

**Versión**: 1.0.0  
**Última actualización**: 22 de enero de 2026  
**Mantenedor**: Cloud Engineering Team
