"""
Módulo para generación de embeddings usando modelos preentrenados.
Utiliza torchvision para modelos ResNet.
"""

import os
import logging
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Modelo para generar embeddings de imágenes."""

    AVAILABLE_MODELS = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    def __init__(self, model_name: str = "resnet50", use_gpu: bool = True):
        """
        Inicializa el modelo de embeddings.
        
        Args:
            model_name: Nombre del modelo (resnet50, resnet101, resnet152)
            use_gpu: Usar GPU si está disponible
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Modelo no soportado: {model_name}. "
                f"Disponibles: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model_name
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Usando dispositivo: {self.device}")
        logger.info(f"Cargando modelo: {model_name}")

        # Cargar modelo preentrenado
        model_fn = self.AVAILABLE_MODELS[model_name]
        self.model = model_fn(weights="DEFAULT")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Crear extractor de features (remover clasificador final)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()

        # Transformaciones normalizadas para ImageNet
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info(f"Modelo {model_name} listo para generar embeddings")

    def get_embedding(self, image_array: np.ndarray) -> np.ndarray:
        """
        Genera embedding para una imagen.
        
        Args:
            image_array: Array de numpy en formato RGB (height, width, 3)
            
        Returns:
            Array de embedding (1, 2048) para ResNet50
        """
        try:
            # Aplicar transformaciones
            image_tensor = self.transforms(image_array)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Generar embedding
            with torch.no_grad():
                embedding = self.feature_extractor(image_tensor)

            # Convertir a numpy y aplanar
            embedding_np = embedding.cpu().numpy().flatten()

            return embedding_np

        except Exception as e:
            logger.error(f"Error generando embedding: {str(e)}")
            raise

    def get_batch_embeddings(self, image_arrays: list) -> np.ndarray:
        """
        Genera embeddings para múltiples imágenes.
        
        Args:
            image_arrays: Lista de arrays de numpy
            
        Returns:
            Array de embeddings (N, 2048)
        """
        embeddings = []

        for idx, image_array in enumerate(image_arrays):
            try:
                embedding = self.get_embedding(image_array)
                embeddings.append(embedding)
                logger.debug(f"Embedding {idx + 1}/{len(image_arrays)} generado")
            except Exception as e:
                logger.error(f"Error en imagen {idx}: {str(e)}")
                raise

        return np.array(embeddings)

    def aggregate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Agrega embeddings múltiples usando promedio.
        
        Args:
            embeddings: Array de embeddings (N, embedding_dim)
            
        Returns:
            Array promedio (embedding_dim,)
        """
        return np.mean(embeddings, axis=0)


def get_embedding_model(use_gpu: bool = True) -> EmbeddingModel:
    """
    Factory para crear el modelo de embeddings.
    Usa variable de entorno MODEL_NAME si está disponible.
    
    Args:
        use_gpu: Usar GPU si está disponible
        
    Returns:
        Instancia de EmbeddingModel
    """
    model_name = os.getenv("MODEL_NAME", "resnet50")
    return EmbeddingModel(model_name=model_name, use_gpu=use_gpu)
