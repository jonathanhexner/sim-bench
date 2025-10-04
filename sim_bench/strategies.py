"""
Distance/similarity strategies for sim-bench.
Implements Strategy pattern for different distance computations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class DistanceStrategy(ABC):
    """Abstract strategy for computing distances between feature vectors."""
    
    @abstractmethod
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between two feature matrices.
        
        Args:
            features_a: Feature matrix [n_samples_a, n_features]
            features_b: Feature matrix [n_samples_b, n_features]
            
        Returns:
            Distance matrix [n_samples_a, n_samples_b]
        """
        pass


class ChiSquareStrategy(DistanceStrategy):
    """Chi-square distance strategy for histograms."""
    
    def __init__(self, eps: float = 1e-10):
        self.eps = eps
    
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """Compute chi-square distances between histograms."""
        n, d = features_a.shape
        m, _ = features_b.shape
        D = np.empty((n, m), dtype=np.float32)
        
        # Chunked computation for memory efficiency
        chunk = max(1, 1024 // max(1, d // 256))
        for i in range(0, n, chunk):
            a = features_a[i:i+chunk][:, None, :]  # [c,1,d]
            b = features_b[None, :, :]             # [1,m,d]
            num = (a - b) ** 2
            den = (a + b + self.eps)
            chi = 0.5 * np.sum(num / den, axis=2)  # [c,m]
            D[i:i+chunk] = chi
        return D


class CosineStrategy(DistanceStrategy):
    """Cosine distance strategy (1 - cosine_similarity)."""
    
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """Compute cosine distances between feature vectors."""
        # L2 normalize
        a_norm = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-12)
        b_norm = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-12)
        
        # Cosine similarity then convert to distance
        similarity = a_norm @ b_norm.T
        return 1.0 - similarity


class EuclideanStrategy(DistanceStrategy):
    """Euclidean distance strategy."""
    
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between feature vectors."""
        # Using broadcasting: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        sq_a = np.sum(features_a * features_a, axis=1, keepdims=True)  # [n,1]
        sq_b = np.sum(features_b * features_b, axis=1, keepdims=True).T  # [1,m]
        dot_ab = features_a @ features_b.T  # [n,m]
        
        distances = sq_a + sq_b - 2 * dot_ab
        return np.maximum(distances, 0.0)  # Ensure non-negative


class WassersteinStrategy(DistanceStrategy):
    """Earth Mover's Distance (Wasserstein) strategy for histograms."""
    
    def __init__(self, ground_metric: str = 'euclidean_xyz'):
        self.ground_metric = ground_metric
        
    def compute_pairwise_distances(self, features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
        """Compute Wasserstein distances between histograms."""
        from scipy.stats import wasserstein_distance
        
        n, d = features_a.shape
        m, _ = features_b.shape
        D = np.empty((n, m), dtype=np.float32)
        
        # Create coordinate arrays for histogram bins
        # For HSV histograms with shape (h_bins, s_bins, v_bins), create 3D coordinates
        if d == 4096:  # 16x16x16 HSV histogram
            h_bins, s_bins, v_bins = 16, 16, 16
        elif d == 512:  # 8x8x8 HSV histogram  
            h_bins, s_bins, v_bins = 8, 8, 8
        else:
            # Generic case: assume 1D histogram
            coords = np.arange(d)
            
            for i in range(n):
                for j in range(m):
                    D[i, j] = wasserstein_distance(coords, coords, features_a[i], features_b[j])
            return D
        
        # Create 3D coordinate grid for HSV histograms
        coords = []
        for h in range(h_bins):
            for s in range(s_bins):
                for v in range(v_bins):
                    coords.append([h, s, v])
        coords = np.array(coords)
        
        # Compute pairwise Wasserstein distances
        for i in range(n):
            for j in range(m):
                D[i, j] = wasserstein_distance(coords[:, 0], coords[:, 0], features_a[i], features_b[j])
        
        return D


def create_distance_strategy(strategy_name: str, **kwargs) -> DistanceStrategy:
    """
    Factory function for distance strategies.
    
    Args:
        strategy_name: Name of the distance strategy
        **kwargs: Strategy-specific parameters
        
    Returns:
        Instantiated distance strategy
        
    Raises:
        ValueError: If strategy_name is not recognized
    """
    if strategy_name == 'chi_square' or strategy_name == 'chi2':
        return ChiSquareStrategy(**kwargs)
    elif strategy_name == 'cosine':
        return CosineStrategy(**kwargs)
    elif strategy_name == 'euclidean':
        return EuclideanStrategy(**kwargs)
    elif strategy_name == 'wasserstein' or strategy_name == 'emd':
        return WassersteinStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown distance strategy: {strategy_name}. "
                        f"Available: chi_square, cosine, euclidean, wasserstein")
