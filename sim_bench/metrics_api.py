"""
Metrics computation API using proper factory pattern.
Entry point that delegates to the MetricFactory for clean architecture.
"""

# Import the factory class from the metrics package
from sim_bench.metrics import MetricFactory

# Main entry point - delegates to factory
def compute_metrics(ranking_indices, evaluation_data, config):
    """
    Compute metrics using the factory pattern.
    
    Args:
        ranking_indices: Sorted ranking indices [n_queries, n_gallery]
        evaluation_data: Dataset evaluation data (groups, relevance_map, etc.)
        config: Configuration specifying which metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    return MetricFactory.compute_all_metrics(ranking_indices, evaluation_data, config)