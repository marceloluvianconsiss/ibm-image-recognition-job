"""
Orquestador principal del Job de reconocimiento de imágenes en IBM Code Engine.

Workflow:
1. Conectar a COS
2. Listar imágenes organizadas por producto
3. Para cada producto:
   - Cargar todas sus imágenes
   - Validar y convertir a RGB
   - Generar embeddings
   - Agrupar embeddings (promedio)
   - Guardar resultado en COS
4. Ser idempotente (no procesar si ya existen embeddings)
"""

import logging
import sys
import os
from collections import defaultdict

from cos_client import COSClient
from image_utils import ImageValidator
from embedding_model import get_embedding_model


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def validate_environment():
    """Valida que las variables de entorno obligatorias estén configuradas."""
    required_vars = [
        "COS_ENDPOINT",
        "COS_ACCESS_KEY_ID",
        "COS_SECRET_ACCESS_KEY",
        "COS_BUCKET_NAME",
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Faltan variables de entorno obligatorias: {', '.join(missing)}")
    
    logger.info("Variables de entorno validadas")


def process_product_images(
    cos_client: COSClient,
    embedding_model,
    product_sku: str,
    image_keys: list,
) -> bool:
    """
    Procesa todas las imágenes de un producto.
    
    Args:
        cos_client: Cliente de COS
        embedding_model: Modelo para generar embeddings
        product_sku: SKU del producto
        image_keys: Lista de claves COS de imágenes del producto
        
    Returns:
        True si se procesó correctamente, False en caso contrario
    """
    try:
        logger.info(f"Procesando producto: {product_sku} ({len(image_keys)} imágenes)")

        # Cargar todas las imágenes válidas
        valid_images = []
        image_filenames = []

        for image_key in image_keys:
            filename = image_key.split("/")[-1]

            try:
                # Leer imagen de COS
                image_data = cos_client.read_image(image_key)

                # Validar imagen
                if not ImageValidator.validate_image(filename, image_data):
                    logger.warning(f"Imagen inválida: {image_key}")
                    continue

                # Cargar como RGB
                image_array = ImageValidator.load_image_as_rgb(filename, image_data)
                valid_images.append(image_array)
                image_filenames.append(filename)

                logger.debug(f"Imagen cargada: {filename}")

            except Exception as e:
                logger.error(f"Error procesando {image_key}: {str(e)}")
                continue

        if not valid_images:
            logger.warning(f"No hay imágenes válidas para {product_sku}")
            return False

        # Generar embeddings
        logger.info(f"Generando embeddings para {len(valid_images)} imágenes...")
        embeddings = embedding_model.get_batch_embeddings(valid_images)

        # Agregar embeddings (promedio)
        aggregated_embedding = embedding_model.aggregate_embeddings(embeddings)
        logger.info(f"Embeddings agregados para {product_sku}: shape {aggregated_embedding.shape}")

        # Guardar en COS (ambos formatos para flexibilidad)
        cos_client.save_embeddings_npy(product_sku, aggregated_embedding)
        cos_client.save_embeddings_json(product_sku, aggregated_embedding)

        logger.info(f"✓ Producto completado: {product_sku}")
        return True

    except Exception as e:
        logger.error(f"Error procesando producto {product_sku}: {str(e)}")
        return False


def main():
    """Función principal del Job."""
    try:
        logger.info("=" * 60)
        logger.info("Iniciando Job: IBM Image Recognition")
        logger.info("=" * 60)

        # Validar variables de entorno
        validate_environment()

        # Inicializar cliente COS
        logger.info("Conectando a IBM Cloud Object Storage...")
        cos_client = COSClient()

        # Inicializar modelo
        logger.info("Cargando modelo de embeddings...")
        embedding_model = get_embedding_model(model_name="resnet50")

        # Listar imágenes
        logger.info("Listando imágenes en COS...")
        images = cos_client.list_product_images()

        if not images:
            logger.warning("No se encontraron imágenes. Job finalizado.")
            return 0

        # Agrupar por producto
        products = defaultdict(list)
        for product_sku, image_key in images:
            products[product_sku].append(image_key)

        logger.info(f"Encontrados {len(products)} productos")

        # Procesar cada producto
        successful = 0
        failed = 0

        for product_sku in sorted(products.keys()):
            image_keys = products[product_sku]

            # Verificar idempotencia
            if cos_client.check_embeddings_exist(product_sku, format="npy"):
                logger.info(f"Embeddings ya existen para {product_sku}, omitiendo...")
                successful += 1
                continue

            # Procesar producto
            if process_product_images(
                cos_client,
                embedding_model,
                product_sku,
                image_keys,
            ):
                successful += 1
            else:
                failed += 1

        # Resumen
        logger.info("=" * 60)
        logger.info(f"Job completado:")
        logger.info(f"  - Productos procesados: {successful}")
        logger.info(f"  - Errores: {failed}")
        logger.info("=" * 60)

        return 0 if failed == 0 else 1

    except Exception as e:
        logger.error(f"Error crítico en Job: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
