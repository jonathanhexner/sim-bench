"""
Precision@k metric implementation.
"""

import numpy as np
from typing import List, Set
from .base import BaseMetric


class PrecisionAtK(BaseMetric):
    """Precision@k - average precision in top-k results."""
    
    def __init__(self, k: int = 10, **kwargs):
        """
        Initialize Precision@k metric.
        
        Args:
            k: Number of top results to consider
        """
        super().__init__(**kwargs)
        self.k = k
    
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> float:
        """
        Compute Precision@k.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Average Precision@k score (0.0 to 1.0)
        """
        precision_sum = 0.0
        n_queries = len(relevance_sets)
        
        for query_idx in range(n_queries):
            relevant_images = relevance_sets[query_idx]
            if not relevant_images:
                continue
                
            # Count relevant images in top-k (excluding self at rank 0)
            topk_results = ranking_indices[query_idx][1:self.k+1]
            relevant_in_topk = sum(1 for idx in topk_results if idx in relevant_images)
            
            precision_sum += relevant_in_topk / min(self.k, len(topk_results))
        
        return precision_sum / n_queries if n_queries > 0 else 0.0
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return f"precision@{self.k}"
    
    @property
    def description(self) -> str:
        """Get metric description."""
        return f"Average precision in top-{self.k} results"
