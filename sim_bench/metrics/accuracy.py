"""
Accuracy metric implementation.
"""

import numpy as np
from typing import List, Set
from .base import BaseMetric


class Accuracy(BaseMetric):
    """Accuracy - fraction of queries with at least one relevant result in top-1."""
    
    def __init__(self, **kwargs):
        """Initialize Accuracy metric."""
        super().__init__(**kwargs)
    
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> float:
        """
        Compute Accuracy (same as Recall@1).
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        hits = 0
        n_queries = len(relevance_sets)
        
        for query_idx in range(n_queries):
            relevant_images = relevance_sets[query_idx]
            if not relevant_images:
                continue
                
            # Check if top-1 result is relevant (excluding self at rank 0)
            if len(ranking_indices[query_idx]) > 1:
                top1_result = ranking_indices[query_idx][1]
                if top1_result in relevant_images:
                    hits += 1
        
        return hits / n_queries if n_queries > 0 else 0.0
    
    @property
    def name(self) -> str:
        """Get metric name."""
        return "accuracy"
    
    @property
    def description(self) -> str:
        """Get metric description."""
        return "Fraction of queries with relevant result at rank 1 (same as Recall@1)"
