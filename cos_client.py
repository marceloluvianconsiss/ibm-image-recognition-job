"""
Módulo para interactuar con IBM Cloud Object Storage (COS).
Maneja conexión, listado, lectura y escritura de objetos.
"""

import os
import logging
from typing import List, Tuple, Iterator
from ibm_cos_sdk import Config, ServiceInstance
from ibm_cos_sdk.auth import IAMAuthenticator
import numpy as np
import json

logger = logging.getLogger(__name__)


class COSClient:
    """Cliente para IBM Cloud Object Storage."""

    def __init__(self):
        """Inicializa la conexión a COS usando variables de entorno."""
        self.endpoint = os.getenv("COS_ENDPOINT")
        self.access_key = os.getenv("COS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("COS_SECRET_ACCESS_KEY")
        self.bucket = os.getenv("COS_BUCKET_NAME")
        self.output_prefix = os.getenv("COS_OUTPUT_PREFIX", "embeddings/")

        if not all([self.endpoint, self.access_key, self.secret_key, self.bucket]):
            raise ValueError(
                "Faltan variables de entorno: COS_ENDPOINT, COS_ACCESS_KEY_ID, "
                "COS_SECRET_ACCESS_KEY, COS_BUCKET_NAME"
            )

        # Configurar autenticador
        authenticator = IAMAuthenticator(apikey=self.secret_key)

        # Configurar cliente COS
        config = Config(authenticator=authenticator)
        self.cos_client = ServiceInstance(
            service_name="cos",
            config=config,
            endpoint_url=self.endpoint,
        )

        logger.info(f"Conectado a COS: {self.bucket}")

    def list_product_images(self) -> List[Tuple[str, str]]:
        """
        Lista todas las imágenes organizadas por producto.
        
        Retorna lista de tuplas (producto_sku, ruta_objeto).
        Ejemplo: [('producto-SKU001', 'producto-SKU001/img_01.webp'), ...]
        """
        images = []
        try:
            # Listar todos los objetos en el bucket
            response = self.cos_client.list_objects(Bucket=self.bucket)

            if "Contents" not in response:
                logger.warning("El bucket está vacío")
                return images

            # Agrupar por producto
            for obj in response["Contents"]:
                key = obj["Key"]

                # Validar estructura: producto-<sku>/imagen.ext
                if "/" not in key:
                    logger.debug(f"Ignorando objeto sin estructura de producto: {key}")
                    continue

                parts = key.split("/")
                if len(parts) < 2:
                    continue

                product_sku = parts[0]
                filename = parts[-1]

                # Validar que sea una imagen válida
                valid_extensions = (".webp", ".jpg", ".jpeg", ".png")
                if not filename.lower().endswith(valid_extensions):
                    logger.debug(f"Ignorando archivo no-imagen: {key}")
                    continue

                images.append((product_sku, key))

            logger.info(f"Encontradas {len(images)} imágenes")
            return images

        except Exception as e:
            logger.error(f"Error listando objetos de COS: {str(e)}")
            raise

    def read_image(self, object_key: str) -> bytes:
        """
        Lee una imagen desde COS.
        
        Args:
            object_key: Clave del objeto en COS
            
        Returns:
            Bytes de la imagen
        """
        try:
            response = self.cos_client.get_object(Bucket=self.bucket, Key=object_key)
            image_data = response["Body"].read()
            logger.debug(f"Imagen leída: {object_key} ({len(image_data)} bytes)")
            return image_data
        except Exception as e:
            logger.error(f"Error leyendo imagen {object_key}: {str(e)}")
            raise

    def save_embeddings_npy(self, product_sku: str, embeddings: np.ndarray) -> str:
        """
        Guarda embeddings en formato numpy.
        
        Args:
            product_sku: SKU del producto
            embeddings: Array de numpy con embeddings
            
        Returns:
            Clave del objeto guardado en COS
        """
        try:
            # Crear clave: embeddings/producto-SKU/embeddings.npy
            output_key = f"{self.output_prefix}{product_sku}/embeddings.npy"

            # Convertir numpy array a bytes
            embedding_bytes = embeddings.tobytes()

            # Guardar en COS
            self.cos_client.put_object(
                Bucket=self.bucket,
                Key=output_key,
                Body=embedding_bytes,
                Metadata={
                    "shape": str(embeddings.shape),
                    "dtype": str(embeddings.dtype),
                },
            )

            logger.info(f"Embeddings guardados: {output_key}")
            return output_key

        except Exception as e:
            logger.error(f"Error guardando embeddings para {product_sku}: {str(e)}")
            raise

    def save_embeddings_json(self, product_sku: str, embeddings: np.ndarray) -> str:
        """
        Guarda embeddings en formato JSON.
        
        Args:
            product_sku: SKU del producto
            embeddings: Array de numpy con embeddings
            
        Returns:
            Clave del objeto guardado en COS
        """
        try:
            output_key = f"{self.output_prefix}{product_sku}/embeddings.json"

            # Convertir a lista y guardar como JSON
            data = {
                "product_sku": product_sku,
                "embeddings": embeddings.tolist(),
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
            }

            json_str = json.dumps(data)

            self.cos_client.put_object(
                Bucket=self.bucket,
                Key=output_key,
                Body=json_str.encode("utf-8"),
                ContentType="application/json",
            )

            logger.info(f"Embeddings JSON guardados: {output_key}")
            return output_key

        except Exception as e:
            logger.error(f"Error guardando JSON para {product_sku}: {str(e)}")
            raise

    def check_embeddings_exist(self, product_sku: str, format: str = "npy") -> bool:
        """
        Verifica si ya existen embeddings para un producto (idempotencia).
        
        Args:
            product_sku: SKU del producto
            format: 'npy' o 'json'
            
        Returns:
            True si existen, False en caso contrario
        """
        try:
            extension = "npy" if format == "npy" else "json"
            embedding_key = f"{self.output_prefix}{product_sku}/embeddings.{extension}"

            response = self.cos_client.head_object(
                Bucket=self.bucket,
                Key=embedding_key,
            )
            return True
        except self.cos_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
        except Exception as e:
            logger.debug(f"Error verificando existencia de embeddings: {str(e)}")
            return False
