"""
Módulo para validación y carga de imágenes.
Soporta formatos: webp, jpg, jpeg, png.
"""

import logging
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validador y cargador de imágenes."""

    SUPPORTED_FORMATS = {".webp", ".jpg", ".jpeg", ".png"}

    @staticmethod
    def get_image_format(filename: str) -> Optional[str]:
        """
        Detecta el formato de imagen.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Formato detectado (.webp, .jpg, etc.) o None si no es válido
        """
        filename_lower = filename.lower()
        for fmt in ImageValidator.SUPPORTED_FORMATS:
            if filename_lower.endswith(fmt):
                return fmt
        return None

    @staticmethod
    def validate_image(filename: str, image_data: bytes) -> bool:
        """
        Valida que el archivo sea una imagen soportada.
        
        Args:
            filename: Nombre del archivo
            image_data: Bytes de la imagen
            
        Returns:
            True si es válida, False en caso contrario
        """
        try:
            # Verificar extensión
            if not ImageValidator.get_image_format(filename):
                logger.warning(f"Formato no soportado: {filename}")
                return False

            # Intentar abrir la imagen
            img = Image.open(BytesIO(image_data))
            img.verify()
            logger.debug(f"Imagen válida: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error validando imagen {filename}: {str(e)}")
            return False

    @staticmethod
    def load_image_as_rgb(filename: str, image_data: bytes) -> np.ndarray:
        """
        Carga una imagen y la convierte a RGB (si es necesario).
        
        Args:
            filename: Nombre del archivo
            image_data: Bytes de la imagen
            
        Returns:
            Array de numpy en formato RGB (height, width, 3)
            
        Raises:
            ValueError: Si la imagen no puede ser cargada
        """
        try:
            # Abrir imagen
            img = Image.open(BytesIO(image_data))

            # Convertir a RGB si es necesario
            if img.mode != "RGB":
                logger.debug(
                    f"Convirtiendo {filename} de modo {img.mode} a RGB"
                )
                img = img.convert("RGB")

            # Convertir a numpy array
            img_array = np.array(img)

            logger.debug(
                f"Imagen cargada: {filename}, shape: {img_array.shape}"
            )
            return img_array

        except Exception as e:
            logger.error(f"Error cargando imagen {filename}: {str(e)}")
            raise ValueError(f"No se puede cargar imagen {filename}: {str(e)}")

    @staticmethod
    def get_image_info(filename: str, image_data: bytes) -> dict:
        """
        Obtiene información de la imagen.
        
        Args:
            filename: Nombre del archivo
            image_data: Bytes de la imagen
            
        Returns:
            Diccionario con info: formato, tamaño, modo, dimensiones
        """
        try:
            img = Image.open(BytesIO(image_data))
            return {
                "filename": filename,
                "format": img.format,
                "size": img.size,  # (width, height)
                "mode": img.mode,
                "bytes": len(image_data),
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de {filename}: {str(e)}")
            return {"filename": filename, "error": str(e)}
