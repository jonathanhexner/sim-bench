"""
Proper factory for metrics creation.
Encapsulates all metric creation logic and hides concrete implementations.
"""

import numpy as np
from typing import Dict, List, Set, Any, Union
from .base import BaseMetric, create_relevance_sets_from_groups, create_relevance_sets_from_map


class MetricFactory:
    """
    Factory for creating and managing metrics.
    Provides a clean interface without exposing concrete metric classes.
    """
    
    @staticmethod
    def create_metric(metric_name: str, **kwargs) -> BaseMetric:
        """
        Create a metric instance by name.
        
        Args:
            metric_name: Name of the metric to create
            **kwargs: Metric-specific parameters
            
        Returns:
            Metric instance
            
        Raises:
            ValueError: If metric name is not recognized
        """
        # Import concrete classes only when needed (lazy loading)
        # Filter out 'k' from kwargs to avoid duplicate keyword arguments
        filtered_kwargs = {key: value for key, value in kwargs.items() if key != 'k'}
        
        if metric_name.startswith('recall@'):
            from .recall import RecallAtK
            k = int(metric_name.split('@')[1])
            return RecallAtK(k=k, **filtered_kwargs)
        
        elif metric_name.startswith('precision@'):
            from .precision import PrecisionAtK
            k = int(metric_name.split('@')[1])
            return PrecisionAtK(k=k, **filtered_kwargs)
        
        elif metric_name == 'map' or metric_name == 'map_full':
            from .average_precision import MeanAveragePrecision
            return MeanAveragePrecision(k=None, **filtered_kwargs)
        
        elif metric_name.startswith('map@'):
            from .average_precision import MeanAveragePrecision
            k = int(metric_name.split('@')[1])
            return MeanAveragePrecision(k=k, **filtered_kwargs)
        
        elif metric_name == 'ns' or metric_name == 'ns_score':
            from .normalized_score import NormalizedScore
            k = kwargs.get('k', 4)
            return NormalizedScore(k=k, **filtered_kwargs)
        
        elif metric_name == 'accuracy':
            from .accuracy import Accuracy
            return Accuracy(**kwargs)
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}. "
                           f"Available: recall@k, precision@k, map[@k], ns_score, accuracy")
    
    @staticmethod
    def get_available_metrics() -> List[str]:
        """
        Get list of all available metric types.
        
        Returns:
            List of available metric names/patterns
        """
        return [
            'recall@k',      # e.g., recall@1, recall@4, recall@10
            'precision@k',   # e.g., precision@10, precision@50
            'map',           # Mean Average Precision (full)
            'map@k',         # e.g., map@10, map@50
            'ns_score',      # Normalized Score
            'accuracy'       # Accuracy (same as Recall@1)
        ]
    
    @staticmethod
    def compute_all_metrics(ranking_indices: np.ndarray, evaluation_data: Dict[str, Any], 
                           config: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all requested metrics using the factory.
        
        Args:
            ranking_indices: Sorted ranking indices [n_queries, n_gallery]
            evaluation_data: Dataset evaluation data (groups, relevance_map, etc.)
            config: Configuration specifying which metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert evaluation data to universal relevance sets
        if 'groups' in evaluation_data:
            # UKBench-style: group labels
            relevance_sets = create_relevance_sets_from_groups(evaluation_data['groups'])
        elif 'relevance_map' in evaluation_data:
            # Holidays-style: explicit relevance map
            relevance_sets = create_relevance_sets_from_map(
                evaluation_data['relevance_map'], 
                evaluation_data['queries']
            )
        else:
            raise ValueError("Evaluation data must contain either 'groups' or 'relevance_map'")
        
        results = {}
        requested_metrics = config.get('metrics', [])
        
        # Create and compute each requested metric using factory
        for metric_name in requested_metrics:
            try:
                metric = MetricFactory.create_metric(metric_name, **config)
                value = metric.compute(ranking_indices, relevance_sets)
                results[metric_name] = value
            except Exception as e:
                print(f"Warning: Failed to compute {metric_name}: {e}")
                continue
        
        # Add metadata
        results['num_queries'] = len(relevance_sets)
        if 'total_images' in evaluation_data:
            results['num_images'] = evaluation_data['total_images']
        
        return results
    
    @staticmethod
    def describe_metric(metric_name: str) -> str:
        """
        Get description of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Human-readable description of the metric
        """
        try:
            metric = MetricFactory.create_metric(metric_name)
            return metric.description
        except ValueError:
            return f"Unknown metric: {metric_name}"
