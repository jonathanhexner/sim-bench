"""
Mean Average Precision (mAP) metric implementation.
"""

import numpy as np
from typing import List, Set, Optional
from .base import BaseMetric


class MeanAveragePrecision(BaseMetric):
    """Mean Average Precision (mAP) metric."""
    
    def __init__(self, k: Optional[int] = None, **kwargs):
        """
        Initialize mAP metric.
        
        Args:
            k: Maximum rank to consider (None = use all relevant images)
        """
        super().__init__(**kwargs)
        self.k = k
    
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> float:
        """
        Compute Mean Average Precision.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Mean Average Precision score (0.0 to 1.0)
        """
        ap_scores = self._compute_average_precision_per_query(ranking_indices, relevance_sets)
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def _compute_average_precision_per_query(self, ranking_indices: np.ndarray, 
                                           relevance_sets: List[Set[int]]) -> List[float]:
        """
        Compute Average Precision for each query.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            List of AP scores for each query
        """
        ap_scores = []
        
        for query_idx in range(len(relevance_sets)):
            relevant_images = relevance_sets[query_idx]
            if not relevant_images:
                ap_scores.append(0.0)
                continue
                
            # Determine evaluation depth
            eval_k = self.k if self.k is not None else len(relevant_images)
            eval_k = min(eval_k, len(ranking_indices[query_idx]) - 1)  # -1 to skip self
            
            # Calculate AP for this query
            relevant_count = 0
            precision_sum = 0.0
            
            for rank, result_idx in enumerate(ranking_indices[query_idx][1:eval_k+1], start=1):
                if result_idx in relevant_images:
                    relevant_count += 1
                    precision_sum += relevant_count / rank
            
            # Normalize by number of relevant images
            ap = precision_sum / len(relevant_images) if relevant_images else 0.0
            ap_scores.append(ap)
        
        return ap_scores
    
    @property
    def name(self) -> str:
        """Get metric name."""
        if self.k is None:
            return "map"
        else:
            return f"map@{self.k}"
    
    @property
    def description(self) -> str:
        """Get metric description."""
        if self.k is None:
            return "Mean Average Precision using all relevant images"
        else:
            return f"Mean Average Precision at {self.k}"
