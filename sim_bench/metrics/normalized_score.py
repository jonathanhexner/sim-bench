"""
Normalized Score (N-S) metric implementation.
"""

import numpy as np
from typing import List, Set
from .base import BaseMetric


class NormalizedScore(BaseMetric):
    """Normalized Score (N-S) - average number of relevant images in top-k."""
    
    def __init__(self, k: int = 4, **kwargs):
        """
        Initialize N-S metric.
        
        Args:
            k: Number of top results to consider
        """
        super().__init__(**kwargs)
        self.k = k
    
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> float:
        """
        Compute Normalized Score.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Average number of relevant images found in top-k
        """
        total_hits = 0
        n_queries = len(relevance_sets)
        
        for query_idx in range(n_queries):
            relevant_images = relevance_sets[query_idx]
            if not relevant_images:
                continue
                
            # Count hits in top-k (excluding self at rank 0)
            hits = 0
            for result_idx in ranking_indices[query_idx][1:self.k+1]:
                if result_idx in relevant_images:
                    hits += 1
            
            total_hits += hits
        
        return total_hits / n_queries if n_queries > 0 else 0.0
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return "ns_score"
    
    @property
    def description(self) -> str:
        """Get metric description."""
        return f"Average number of relevant images found in top-{self.k} (Normalized Score)"
