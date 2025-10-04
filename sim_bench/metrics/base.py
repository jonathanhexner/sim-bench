"""
Base metric class and factory for universal metric computation.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Set, Any, Union


class BaseMetric(ABC):
    """Abstract base class for all metrics."""
    
    def __init__(self, **kwargs):
        """Initialize metric with parameters."""
        self.params = kwargs
    
    @abstractmethod
    def compute(self, ranking_indices: np.ndarray, relevance_sets: List[Set[int]]) -> Union[float, List[float]]:
        """
        Compute the metric.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            relevance_sets: List of relevant image sets for each query
            
        Returns:
            Metric value(s)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        pass
    
    @property
    def description(self) -> str:
        """Get metric description."""
        return f"{self.name} metric"


def create_relevance_sets_from_groups(groups: List[int]) -> List[Set[int]]:
    """
    Create relevance sets from group labels (UKBench-style).
    
    Args:
        groups: List of group IDs for each image
        
    Returns:
        List of relevance sets (excluding self)
    """
    relevance_sets = []
    
    for query_idx, query_group in enumerate(groups):
        # Find all images in the same group, excluding the query itself
        relevant_images = {idx for idx, group in enumerate(groups) 
                          if group == query_group and idx != query_idx}
        relevance_sets.append(relevant_images)
    
    return relevance_sets


def create_relevance_sets_from_map(relevance_map: Dict[int, List[int]], 
                                  query_indices: List[int]) -> List[Set[int]]:
    """
    Create relevance sets from relevance map (Holidays-style).
    
    Args:
        relevance_map: Dictionary mapping query indices to lists of relevant image indices
        query_indices: List of query image indices
        
    Returns:
        List of relevance sets
    """
    relevance_sets = []
    
    for query_idx in query_indices:
        relevant_images = set(relevance_map.get(query_idx, []))
        relevance_sets.append(relevant_images)
    
    return relevance_sets


# Removed redundant functions - now handled by MetricFactory
