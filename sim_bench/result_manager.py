"""
Result management for saving experiment outputs.
Handles CSV generation, file organization, and summary creation.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from sim_bench.datasets.base import BaseDataset


class ResultManager:
    """Manages experiment result saving and organization."""
    
    def __init__(self, run_config: Dict[str, Any]):
        """
        Initialize result manager.
        
        Args:
            run_config: Run configuration dictionary
        """
        self.run_config = run_config
        self.run_directory = self._create_run_directory()
    
    def _create_run_directory(self) -> Path:
        """Create timestamped run directory."""
        output_root = Path(self.run_config['output_dir'])
        run_name = self.run_config.get('run_name')
        
        if run_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_directory = output_root / f"{run_name}_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_directory = output_root / timestamp
        
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory
    
    def save_method_results(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        ranking_indices: np.ndarray,
        distance_matrix: np.ndarray,
        computed_metrics: Dict[str, float],
        dataset: BaseDataset
    ) -> None:
        """
        Save complete results for a single method.
        
        Args:
            method_name: Name of the method
            method_config: Method configuration
            ranking_indices: Ranking indices matrix
            distance_matrix: Distance matrix
            computed_metrics: Computed metrics dictionary
            dataset: Dataset instance
        """
        method_directory = self.run_directory / method_name
        method_directory.mkdir(parents=True, exist_ok=True)
        
        # Save different result files
        self._save_metrics_csv(method_directory, method_name, computed_metrics, dataset)
        self._save_per_query_csv(method_directory, ranking_indices, computed_metrics, dataset)
        self._save_rankings_csv(method_directory, ranking_indices, distance_matrix, dataset)
        self._save_manifest_json(method_directory, method_name, method_config)
        
        print(f"Results written to: {method_directory}")
    
    def _save_metrics_csv(
        self,
        method_directory: Path,
        method_name: str,
        computed_metrics: Dict[str, float],
        dataset: BaseDataset
    ) -> None:
        """Save overall metrics CSV."""
        
        if not self.run_config.get('save', {}).get('metrics_csv', True):
            return
        
        metrics_file = method_directory / "metrics.csv"
        
        with open(metrics_file, "w", newline="") as file:
            writer = csv.writer(file)
            
            # Write header and data based on dataset type
            if hasattr(dataset, 'groups'):  # UKBench
                header = ["method", "ns", "recall@1", "recall@4", "map@10", "num_queries", "num_gallery", "created_at"]
                row = [
                    method_name,
                    f"{computed_metrics.get('ns_score', 0):.6f}",
                    f"{computed_metrics.get('recall@1', 0):.6f}",
                    f"{computed_metrics.get('recall@4', 0):.6f}",
                    f"{computed_metrics.get('map@10', 0):.6f}",
                    len(dataset.get_queries()),
                    len(dataset.get_images()),
                    datetime.now().isoformat()
                ]
            else:  # Holidays
                header = ["method", "map_full", "map@10", "map@50", "recall@1", "recall@10", "prec@10", "num_queries", "num_gallery", "created_at"]
                row = [
                    method_name,
                    f"{computed_metrics.get('map_full', 0):.6f}",
                    f"{computed_metrics.get('map@10', 0):.6f}",
                    f"{computed_metrics.get('map@50', 0):.6f}",
                    f"{computed_metrics.get('recall@1', 0):.6f}",
                    f"{computed_metrics.get('recall@10', 0):.6f}",
                    f"{computed_metrics.get('prec@10', 0):.6f}",
                    len(dataset.get_queries()),
                    len(dataset.get_images()),
                    datetime.now().isoformat()
                ]
            
            writer.writerow(header)
            writer.writerow(row)
    
    def _save_per_query_csv(
        self,
        method_directory: Path,
        ranking_indices: np.ndarray,
        computed_metrics: Dict[str, float],
        dataset: BaseDataset
    ) -> None:
        """Save per-query results CSV."""
        
        if not self.run_config.get('save', {}).get('per_query_csv', True):
            return
        
        per_query_file = method_directory / "per_query.csv"
        
        with open(per_query_file, "w", newline="") as file:
            writer = csv.writer(file)
            
            if hasattr(dataset, 'groups'):  # UKBench
                self._save_ukbench_per_query(writer, ranking_indices, dataset)
            else:  # Holidays
                self._save_holidays_per_query(writer, ranking_indices, dataset)
    
    def _save_ukbench_per_query(
        self,
        writer: csv.writer,
        ranking_indices: np.ndarray,
        dataset: BaseDataset
    ) -> None:
        """Save UKBench per-query results."""
        
        writer.writerow(["query_idx", "query_path", "group_id", "ns_hitcount@4", "ap@10"])
        
        images = dataset.get_images()
        groups = dataset.groups
        k_value = self.run_config.get('k', 4)
        
        for query_idx in range(len(images)):
            query_group = groups[query_idx]
            
            # Count hits in top-k (excluding self at rank 0)
            hits = sum(1 for result_idx in ranking_indices[query_idx][1:k_value+1] 
                      if groups[result_idx] == query_group)
            
            # Compute AP@10
            relevant_count = 0
            precision_sum = 0.0
            
            for rank, result_idx in enumerate(ranking_indices[query_idx][1:11], start=1):
                if groups[result_idx] == query_group:
                    relevant_count += 1
                    precision_sum += relevant_count / rank
            
            average_precision_10 = precision_sum / min(10, 3)  # Max 3 relevant per group
            
            writer.writerow([
                query_idx,
                images[query_idx],
                query_group,
                hits,
                f"{average_precision_10:.6f}"
            ])
    
    def _save_holidays_per_query(
        self,
        writer: csv.writer,
        ranking_indices: np.ndarray,
        dataset: BaseDataset
    ) -> None:
        """Save Holidays per-query results."""
        
        writer.writerow(["query_idx", "query_path", "num_relevant", "ap_full", "ap@10", "recall@10"])
        
        images = dataset.get_images()
        queries = dataset.get_queries()
        relevance_map = dataset.data['relevance_map']
        
        for query_idx in queries:
            relevant_images = relevance_map.get(query_idx, [])
            
            # Compute full AP and AP@10
            relevant_count = 0
            precision_sum_full = 0.0
            precision_sum_10 = 0.0
            
            for rank, result_idx in enumerate(ranking_indices[query_idx][1:], start=1):
                if result_idx in relevant_images:
                    relevant_count += 1
                    precision_sum_full += relevant_count / rank
                    if rank <= 10:
                        precision_sum_10 += relevant_count / rank
            
            ap_full = precision_sum_full / len(relevant_images) if relevant_images else 0.0
            ap_10 = precision_sum_10 / min(len(relevant_images), 10) if relevant_images else 0.0
            
            # Compute recall@10
            recall_10 = min(relevant_count, 10) / len(relevant_images) if relevant_images else 0.0
            
            writer.writerow([
                query_idx,
                images[query_idx],
                len(relevant_images),
                f"{ap_full:.6f}",
                f"{ap_10:.6f}",
                f"{recall_10:.6f}"
            ])
    
    def _save_rankings_csv(
        self,
        method_directory: Path,
        ranking_indices: np.ndarray,
        distance_matrix: np.ndarray,
        dataset: BaseDataset
    ) -> None:
        """Save rankings CSV."""
        
        if not self.run_config.get('save', {}).get('rankings_csv', True):
            return
        
        rankings_file = method_directory / "rankings.csv"
        topk_limit = int(self.run_config.get('save', {}).get('topk_rankings_k', 10))
        
        with open(rankings_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["query_idx", "rank", "result_idx", "distance"])
            
            images = dataset.get_images()
            queries = dataset.get_queries()
            
            for query_idx in queries:
                for rank in range(min(topk_limit, len(images))):
                    result_idx = ranking_indices[query_idx][rank]
                    distance = distance_matrix[query_idx][result_idx]
                    writer.writerow([query_idx, rank, result_idx, f"{distance:.6f}"])
    
    def _save_manifest_json(
        self,
        method_directory: Path,
        method_name: str,
        method_config: Dict[str, Any]
    ) -> None:
        """Save method manifest JSON."""
        
        manifest_file = method_directory / "manifest.json"
        manifest_data = {
            "method": method_name,
            "config": method_config,
            "created_at": datetime.now().isoformat()
        }
        
        with open(manifest_file, 'w') as file:
            json.dump(manifest_data, file, indent=2)
    
    def create_summary(self, method_results: List[Dict[str, Any]]) -> None:
        """
        Create summary CSV comparing multiple methods.
        
        Args:
            method_results: List of method result dictionaries
        """
        if not self.run_config.get('save', {}).get('summary_csv', True):
            return
        
        if not method_results:
            return
        
        summary_file = self.run_directory / "summary.csv"
        
        # Flatten results for CSV
        flattened_results = []
        for result in method_results:
            flat_result = {'method': result['method']}
            flat_result.update(result['metrics'])
            flattened_results.append(flat_result)
        
        # Write CSV
        with open(summary_file, 'w', newline='') as file:
            if flattened_results:
                fieldnames = flattened_results[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_results)
        
        print(f"\nSummary saved to: {summary_file}")
