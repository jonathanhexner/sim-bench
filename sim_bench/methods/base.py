"""
Abstract base class for similarity methods in sim-bench.
Uses Strategy pattern for distance computation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from pathlib import Path


class BaseMethod(ABC):
    """Abstract base class for all similarity methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize method with configuration.
        
        Args:
            config: Method configuration dictionary loaded from YAML
        """
        self.config = config
        self.method_name = config['method']
        
        # Initialize distance strategy
        from sim_bench.strategies import create_distance_strategy
        distance_type = config.get('distance', 'cosine')
        self.distance_strategy = create_distance_strategy(distance_type)
        
    @abstractmethod
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from a list of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Feature matrix of shape [n_images, feature_dim]
        """
        pass
    
    def compute_distances(self, features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances using the configured strategy.
        
        Args:
            features: Feature matrix [n_images, feature_dim]
            
        Returns:
            Distance matrix [n_images, n_images]
        """
        return self.distance_strategy.compute_pairwise_distances(features, features)
    
    def get_cache_dir(self) -> Path:
        """Get cache directory for this method."""
        cache_dir = Path(self.config.get('cache_dir', f'artifacts/{self.method_name}'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.method_name})"


def load_method(method_name: str, config: Dict[str, Any]) -> BaseMethod:
    """
    Factory function to load a method by name.
    
    Args:
        method_name: Name of the method to load
        config: Method configuration dictionary
        
    Returns:
        Instantiated method object
        
    Raises:
        ValueError: If method_name is not recognized
    """
    if method_name == 'chi_square':
        from sim_bench.methods.chi_square import ChiSquareMethod
        return ChiSquareMethod(config)
    elif method_name == 'emd':
        from sim_bench.methods.emd import EMDMethod
        return EMDMethod(config)
    elif method_name == 'deep':
        from sim_bench.methods.deep import DeepMethod
        return DeepMethod(config)
    elif method_name == 'sift_bovw':
        from sim_bench.methods.sift_bovw import SIFTBoVWMethod
        return SIFTBoVWMethod(config)
    else:
        raise ValueError(f"Unknown method: {method_name}. Available: chi_square, emd, deep, sift_bovw")
