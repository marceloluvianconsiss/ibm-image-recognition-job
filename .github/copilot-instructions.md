# Instrucciones para Agentes de IA - IBM Image Recognition Job

**Regla Principal**: Responde siempre en español. Actúa como ingeniero senior en cloud, MLOps y backend.

## Arquitectura General

Este Job de IBM Cloud Code Engine procesa imágenes en paralelo desde COS, genera embeddings con ResNet50 y los agrupa por producto. **No es una API HTTP**: es un proceso batch que se ejecuta, procesa y termina.

### Componentes Core

- **`cos_client.py`**: Conexión segura a IBM Cloud Object Storage. Maneja listado, lectura y escritura con reintentos.
- **`image_utils.py`**: Validación de formatos (.webp, .jpg, .jpeg, .png) y conversión a RGB usando Pillow.
- **`embedding_model.py`**: Generación de embeddings con ResNet50 preentrenado (2048 dimensiones) usando PyTorch.
- **`main.py`**: Orquestador que coordina flujo, idempotencia y logging.

### Flujo de Datos

```
COS Bucket (producto-SKU/img.*)
    ↓
list_product_images() → Agrupa por SKU
    ↓
load_image_as_rgb() → Valida formato + convierte RGB
    ↓
get_batch_embeddings() → ResNet50 → (N, 2048)
    ↓
aggregate_embeddings() → Promedio → (2048,)
    ↓
save_embeddings_npy/json() → COS (embeddings/producto-SKU/)
```

## Patrones Clave del Proyecto

### 1. Idempotencia Obligatoria

El Job **NUNCA debe fallar si se ejecuta dos veces**. Ver `main.py:check_embeddings_exist()`:

```python
if cos_client.check_embeddings_exist(product_sku, format="npy"):
    logger.info(f"Embeddings ya existen, omitiendo...")
    continue  # No reprocesar
```

**Patrón**: Antes de cualquier operación costosa, verificar si el output ya existe.

### 2. Configuración Vía Variables de Entorno

**NUNCA hardcodear**. Todos los valores vienen de `os.getenv()`:

```python
# ✓ Correcto
endpoint = os.getenv("COS_ENDPOINT")
model_name = os.getenv("MODEL_NAME", "resnet50")  # con default

# ✗ Incorrecto
endpoint = "https://..."  # Hardcoded
```

Variables esperadas en Code Engine:
- `COS_ENDPOINT`, `COS_ACCESS_KEY_ID`, `COS_SECRET_ACCESS_KEY`, `COS_BUCKET_NAME`
- `COS_OUTPUT_PREFIX` (default: `embeddings/`)
- `MODEL_NAME` (default: `resnet50`)

### 3. Logging Estructurado para Code Engine

Todos los logs van a **stdout** (no stderr) para que Code Engine los capture:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # ← Crítico para Code Engine
)
```

**Niveles**:
- `INFO`: Estados principales, productos completados
- `DEBUG`: Detalles (imágenes procesadas, shapes)
- `WARNING`: Imágenes inválidas (continuar sin fallar)
- `ERROR`: Errores que requieren atención

### 4. Manejo de Errores Resiliente

El Job **continúa con siguiente imagen si una falla**, solo falla el producto si TODAS las imágenes fallan:

```python
# En process_product_images()
for image_key in image_keys:
    try:
        # Procesar imagen
    except Exception as e:
        logger.error(f"Error en {image_key}: {str(e)}")
        continue  # Siguiente imagen

if not valid_images:  # Falla solo si ninguna funcionó
    return False
```

### 5. Integración COS con Reintentos

COS SDK requiere credenciales IAM. Los errores 404 son normales (embeddings no existen aún):

```python
# En cos_client.check_embeddings_exist()
try:
    self.cos_client.head_object(Bucket=..., Key=...)
    return True
except ClientError as e:
    if e.response["Error"]["Code"] == "404":
        return False  # Normal, no existe
    raise  # Otro error, propagar
```

## Extensiones Comunes

### Agregar un Modelo Nuevo

En `embedding_model.py`, modifica `AVAILABLE_MODELS`:

```python
AVAILABLE_MODELS = {
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "vit_b16": models.vision_transformer.vit_b_16,  # Nuevo
}
```

El usuario establece `MODEL_NAME=vit_b16` en Code Engine. Sin cambiar código.

### Cambiar Estrategia de Agregación

En `aggregate_embeddings()` de `embedding_model.py`:

```python
# Default: promedio
return np.mean(embeddings, axis=0)

# Alternativa: máximo (pooling)
return np.max(embeddings, axis=0)

# Alternativa: concatenación
return embeddings.flatten()
```

### Guardar en Formato Adicional

En `cos_client.py`, agregar método como `save_embeddings_parquet()`:

```python
def save_embeddings_parquet(self, product_sku: str, embeddings: np.ndarray):
    import pyarrow.parquet as pq
    table = pa.table({"embeddings": [embeddings.tolist()]})
    # Escribir a bytes y guardar en COS...
```

## Debugging en Code Engine

### Ver Logs de Última Ejecución

```bash
ibmcloud ce jobrun logs --jobrun <nombre> --tail 100
```

### Verificar Variables de Entorno

```bash
ibmcloud ce job get --job image-recognition-job
```

### Simular Localmente

```bash
export COS_ENDPOINT="..."
export COS_ACCESS_KEY_ID="..."
export COS_SECRET_ACCESS_KEY="..."
export COS_BUCKET_NAME="..."

python main.py
```

## Validación Antes de Modificar

Antes de cualquier cambio:

1. **¿Afecta idempotencia?** → Verificar `check_embeddings_exist()`
2. **¿Nueva variable de entorno?** → Documentar en README y usar `os.getenv()` con default
3. **¿Cambio de input/output?** → Actualizar docstrings y ejemplo de estructura COS
4. **¿Modifica logging?** → Usar formato estándar con timestamps
5. **¿Cambia requirements?** → Actualizar `requirements.txt` y `Dockerfile`

## Límites del Sistema

- **Tiempo máximo Code Engine**: Verificar quota regional
- **Memoria**: Default 4GB (modificable con `--memory`)
- **GPU**: No configurada por default (agregar `--gpu 1` si disponible)
- **Tamaño imagen COS**: Máximo 100MB (límite SDK)
- **Dimensión embedding ResNet50**: Siempre 2048 (no configurable)
- **Tiempo de compilación Docker**: ~2-3 minutos (imagen PyTorch precompilada)

## Optimización de Build Time

**IMPORTANTE**: La imagen base `pytorch/pytorch:2.1.1-runtime-ubuntu22.04` incluye PyTorch y TorchVision precompilados.

- **NO instalar torch/torchvision en Dockerfile**: Ya están en la imagen base
- **requirements.txt contiene solo**: ibm-cos-sdk, Pillow, numpy, python-dotenv
- **Reduce tiempo de build** de ~10 minutos a ~2-3 minutos

Si necesitas actualizar PyTorch a una versión diferente, cambiar la imagen base FROM en Dockerfile a una versión compatible de pytorch/pytorch.

## Decisiones Arquitectónicas

| Decisión | Alternativa Rechazada | Por Qué |
|----------|------------------------|--------|
| Imagen base pytorch/pytorch | python:3.11-slim | Evita ~10 minutos compilando PyTorch, reduce timeout en Code Engine |
| Batch processing secuencial | Parallelización | Evitar race conditions en COS y gestión de GPU compleja |
| Promedio de embeddings | Concatenación | Dimensión fija, comparable entre productos |
| Guardado .npy + .json | Solo uno | Flexibilidad: downstream puede elegir formato |
| Factory pattern en `embedding_model.py` | Instancia global | Inyección de `use_gpu` dinámicamente |
| Logging a stdout | Archivo | Code Engine captura stdout automáticamente |

## Archivos Críticos por Tarea

- **"El Job no procesa imágenes"** → `main.py` (listado) + `cos_client.py` (lectura)
- **"Embeddings incorrectos"** → `embedding_model.py` (transformaciones) + `image_utils.py` (RGB)
- **"Falla de conexión COS"** → `cos_client.py` (__init__) + variables de entorno
- **"Idempotencia rota"** → `main.py` (check antes de procesar)
- **"Logs no aparecen"** → `main.py` (stream=sys.stdout)
- **"Dockerfile no construye"** → `Dockerfile` + `requirements.txt` (versiones)

---

**Instrucción Final para Agentes**: Responde siempre en español. Si el usuario pide cambios, sugiere cómo mantener idempotencia e inmutabilidad de variables de entorno. Prioriza claridad de logging para debugging en Code Engine.
