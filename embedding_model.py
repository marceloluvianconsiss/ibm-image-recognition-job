"""
Módulo para generación de embeddings usando modelos preentrenados.
Utiliza torchvision para modelos ResNet.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Modelo para generar embeddings de imágenes."""

    def __init__(self, model_name: str = "resnet50"):
        """
        Inicializa el modelo de embeddings usando CPU únicamente.
        
        Args:
            model_name: Nombre del modelo (resnet50, resnet101, resnet152)
        """
        self.device = torch.device("cpu")
        self.model_name = model_name

        logger.info(f"Cargando modelo: {model_name} en CPU")

        # Cargar modelo preentrenado
        if model_name == "resnet50":
            model = models.resnet50(weights="DEFAULT")
        elif model_name == "resnet101":
            model = models.resnet101(weights="DEFAULT")
        elif model_name == "resnet152":
            model = models.resnet152(weights="DEFAULT")
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")

        self.model = model.to(self.device)
        self.model.eval()

        # Crear extractor de features (remover capa final fc)
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

        logger.info(f"Modelo {model_name} listo (CPU mode)")

    def get_embedding(self, image_array: np.ndarray) -> np.ndarray:
        """
        Genera embedding para una imagen.
        
        Args:
            image_array: Array de numpy en formato RGB (height, width, 3)
            
        Returns:
            Array de embedding (2048,) para ResNet50
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


def get_embedding_model(model_name: str = "resnet50") -> EmbeddingModel:
    """
    Factory para crear el modelo de embeddings.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        Instancia de EmbeddingModel
    """
    return EmbeddingModel(model_name=model_name)
