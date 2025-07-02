"""
Evaluation and analysis module for PU Learning + GNN for plant-pollinator interactions.

This module provides comprehensive evaluation metrics and analysis tools
specifically designed for ecological network prediction tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    dcg_score,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class PUEvaluator:
    """
    Evaluator for PU Learning with plant-pollinator interactions.

    Provides comprehensive evaluation metrics and analysis tools
    for ecological network prediction tasks.
    """

    def __init__(self, model, feature_engineer, plant_to_idx, pollinator_to_idx):
        """
        Initialize evaluator.

        Args:
            model: Trained BipartiteGNN model
            feature_engineer: Fitted PUFeatureEngineer
            plant_to_idx: Plant to index mapping
            pollinator_to_idx: Pollinator to index mapping
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.plant_to_idx = plant_to_idx
        self.pollinator_to_idx = pollinator_to_idx
        self.idx_to_plant = {v: k for k, v in plant_to_idx.items()}
        self.idx_to_pollinator = {v: k for k, v in pollinator_to_idx.items()}

    def evaluate_ranking_metrics(
        self,
        test_positive_pairs: List[Tuple[int, int]],
        test_unlabeled_pairs: List[Tuple[int, int]],
        graph_data,
        k_values: List[int] = [10, 50, 100],
    ) -> Dict[str, float]:
        """
        Evaluate ranking metrics for PU learning.

        Args:
            test_positive_pairs: Test positive pairs
            test_unlabeled_pairs: Test unlabeled pairs
            graph_data: Graph data
            k_values: Values of k for precision@k and recall@k

        Returns:
            Dictionary with ranking metrics
        """
        logger.info("Computing ranking metrics...")

        self.model.eval()
        with torch.no_grad():
            # Get node embeddings
            node_embeddings = self.model(graph_data)

            # Prepare test data
            all_pairs = test_positive_pairs + test_unlabeled_pairs
            labels = [1] * len(test_positive_pairs) + [0] * len(test_unlabeled_pairs)

            pair_tensor = torch.tensor(all_pairs, dtype=torch.long)
            labels_tensor = torch.tensor(labels, dtype=torch.float)

            # Get embeddings for pairs
            plant_embs = node_embeddings[pair_tensor[:, 0]]
            pollinator_embs = node_embeddings[pair_tensor[:, 1]]

            # Predict interactions
            logits = self.model.predict_interaction(
                plant_embs, pollinator_embs
            ).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

            # Basic metrics
            auc = roc_auc_score(labels, probs)
            ap = average_precision_score(labels, probs)

            # Ranking metrics
            metrics = {"auc": auc, "average_precision": ap}

            # Precision@k and Recall@k
            for k in k_values:
                # Sort by probability
                sorted_indices = np.argsort(probs)[::-1]
                top_k_indices = sorted_indices[:k]

                # Precision@k
                precision_at_k = np.mean([labels[i] for i in top_k_indices])
                metrics[f"precision_at_{k}"] = precision_at_k

                # Recall@k
                num_positives = sum(labels)
                recall_at_k = (
                    sum(labels[i] for i in top_k_indices) / num_positives
                    if num_positives > 0
                    else 0
                )
                metrics[f"recall_at_{k}"] = recall_at_k

            # NDCG@k
            for k in k_values:
                try:
                    ndcg_at_k = ndcg_score([labels], [probs], k=k)
                    metrics[f"ndcg_at_{k}"] = ndcg_at_k
                except:
                    metrics[f"ndcg_at_{k}"] = 0.0

            # MRR (Mean Reciprocal Rank)
            mrr = self._compute_mrr(probs, labels)
            metrics["mrr"] = mrr

            # MAP (Mean Average Precision)
            map_score = self._compute_map(probs, labels)
            metrics["map"] = map_score

        logger.info(f"Ranking metrics computed: AUC={auc:.4f}, AP={ap:.4f}")
        return metrics

    def _compute_mrr(self, probs: np.ndarray, labels: List[int]) -> float:
        """Compute Mean Reciprocal Rank."""
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        if not positive_indices:
            return 0.0

        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]

        reciprocal_ranks = []
        for pos_idx in positive_indices:
            rank = np.where(sorted_indices == pos_idx)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)

        return np.mean(reciprocal_ranks)

    def _compute_map(self, probs: np.ndarray, labels: List[int]) -> float:
        """Compute Mean Average Precision."""
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        if not positive_indices:
            return 0.0

        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]

        precisions = []
        num_correct = 0

        for i, idx in enumerate(sorted_indices):
            if labels[idx] == 1:
                num_correct += 1
                precision = num_correct / (i + 1)
                precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def evaluate_ecological_metrics(
        self,
        interactions: pd.DataFrame,
        test_positive_pairs: List[Tuple[int, int]],
        test_unlabeled_pairs: List[Tuple[int, int]],
        graph_data,
    ) -> Dict[str, float]:
        """
        Evaluate ecological-specific metrics.

        Args:
            interactions: Full interaction dataset
            test_positive_pairs: Test positive pairs
            test_unlabeled_pairs: Test unlabeled pairs
            graph_data: Graph data

        Returns:
            Dictionary with ecological metrics
        """
        logger.info("Computing ecological metrics...")

        # Get predictions
        probs = self._get_predictions(
            test_positive_pairs + test_unlabeled_pairs, graph_data
        )

        # Ecological metrics
        metrics = {}

        # 1. Taxonomic consistency
        metrics["taxonomic_consistency"] = self._compute_taxonomic_consistency(
            test_positive_pairs, test_unlabeled_pairs, probs, interactions
        )

        # 2. Temporal consistency
        metrics["temporal_consistency"] = self._compute_temporal_consistency(
            test_positive_pairs, test_unlabeled_pairs, probs, interactions
        )

        # 3. Spatial consistency
        metrics["spatial_consistency"] = self._compute_spatial_consistency(
            test_positive_pairs, test_unlabeled_pairs, probs, interactions
        )

        # 4. Network structure preservation
        metrics["network_structure_preservation"] = (
            self._compute_network_structure_preservation(
                test_positive_pairs, test_unlabeled_pairs, probs, interactions
            )
        )

        logger.info(f"Ecological metrics computed: {metrics}")
        return metrics

    def _get_predictions(self, pairs: List[Tuple[int, int]], graph_data) -> np.ndarray:
        """Get predictions for pairs."""
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(graph_data)
            pair_tensor = torch.tensor(pairs, dtype=torch.long)
            plant_embs = node_embeddings[pair_tensor[:, 0]]
            pollinator_embs = node_embeddings[pair_tensor[:, 1]]
            logits = self.model.predict_interaction(
                plant_embs, pollinator_embs
            ).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def _compute_taxonomic_consistency(
        self,
        positive_pairs: List[Tuple[int, int]],
        unlabeled_pairs: List[Tuple[int, int]],
        probs: np.ndarray,
        interactions: pd.DataFrame,
    ) -> float:
        """Compute taxonomic consistency of predictions."""

        # Get family information
        plant_families = {}
        pollinator_families = {}

        for plant_idx, plant_name in self.idx_to_plant.items():
            plant_data = interactions[interactions["Plant_species"] == plant_name]
            if not plant_data.empty:
                plant_families[plant_idx] = (
                    plant_data["Plant_family"].iloc[0]
                    if "Plant_family" in plant_data.columns
                    else "Unknown"
                )

        for pollinator_idx, pollinator_name in self.idx_to_pollinator.items():
            pollinator_data = interactions[
                interactions["Pollinator_species"] == pollinator_name
            ]
            if not pollinator_data.empty:
                pollinator_families[pollinator_idx] = (
                    pollinator_data["Pollinator_family"].iloc[0]
                    if "Pollinator_family" in pollinator_data.columns
                    else "Unknown"
                )

        # Compute family-level interaction probabilities
        family_interactions = {}
        for i, (plant_idx, pollinator_idx) in enumerate(
            positive_pairs + unlabeled_pairs
        ):
            plant_family = plant_families.get(plant_idx, "Unknown")
            pollinator_family = pollinator_families.get(pollinator_idx, "Unknown")
            family_pair = (plant_family, pollinator_family)

            if family_pair not in family_interactions:
                family_interactions[family_pair] = []
            family_interactions[family_pair].append(probs[i])

        # Compute consistency (variance within families should be low)
        consistencies = []
        for family_pair, probs_list in family_interactions.items():
            if len(probs_list) > 1:
                variance = np.var(probs_list)
                consistencies.append(1.0 / (1.0 + variance))

        return np.mean(consistencies) if consistencies else 0.0

    def _compute_temporal_consistency(
        self,
        positive_pairs: List[Tuple[int, int]],
        unlabeled_pairs: List[Tuple[int, int]],
        probs: np.ndarray,
        interactions: pd.DataFrame,
    ) -> float:
        """Compute temporal consistency of predictions."""

        # Group by month and compute average predictions
        monthly_predictions = {}

        for i, (plant_idx, pollinator_idx) in enumerate(
            positive_pairs + unlabeled_pairs
        ):
            plant_name = self.idx_to_plant[plant_idx]
            pollinator_name = self.idx_to_pollinator[pollinator_idx]

            # Get months where this pair could interact
            plant_months = set(
                interactions[interactions["Plant_species"] == plant_name]["Month"]
            )
            pollinator_months = set(
                interactions[interactions["Pollinator_species"] == pollinator_name][
                    "Month"
                ]
            )
            overlap_months = plant_months.intersection(pollinator_months)

            for month in overlap_months:
                if month not in monthly_predictions:
                    monthly_predictions[month] = []
                monthly_predictions[month].append(probs[i])

        # Compute temporal consistency (predictions should be similar across months)
        if len(monthly_predictions) > 1:
            monthly_means = [
                np.mean(probs_list) for probs_list in monthly_predictions.values()
            ]
            temporal_variance = np.var(monthly_means)
            return 1.0 / (1.0 + temporal_variance)

        return 0.0

    def _compute_spatial_consistency(
        self,
        positive_pairs: List[Tuple[int, int]],
        unlabeled_pairs: List[Tuple[int, int]],
        probs: np.ndarray,
        interactions: pd.DataFrame,
    ) -> float:
        """Compute spatial consistency of predictions."""

        # Group by country and compute average predictions
        country_predictions = {}

        for i, (plant_idx, pollinator_idx) in enumerate(
            positive_pairs + unlabeled_pairs
        ):
            plant_name = self.idx_to_plant[plant_idx]
            pollinator_name = self.idx_to_pollinator[pollinator_idx]

            # Get countries where this pair could interact
            plant_countries = set(
                interactions[interactions["Plant_species"] == plant_name]["Country"]
            )
            pollinator_countries = set(
                interactions[interactions["Pollinator_species"] == pollinator_name][
                    "Country"
                ]
            )
            overlap_countries = plant_countries.intersection(pollinator_countries)

            for country in overlap_countries:
                if country not in country_predictions:
                    country_predictions[country] = []
                country_predictions[country].append(probs[i])

        # Compute spatial consistency
        if len(country_predictions) > 1:
            country_means = [
                np.mean(probs_list) for probs_list in country_predictions.values()
            ]
            spatial_variance = np.var(country_means)
            return 1.0 / (1.0 + spatial_variance)

        return 0.0

    def _compute_network_structure_preservation(
        self,
        positive_pairs: List[Tuple[int, int]],
        unlabeled_pairs: List[Tuple[int, int]],
        probs: np.ndarray,
        interactions: pd.DataFrame,
    ) -> float:
        """Compute network structure preservation."""

        # Create observed network
        observed_network = nx.Graph()
        for _, row in interactions.iterrows():
            observed_network.add_edge(row["Plant_species"], row["Pollinator_species"])

        # Create predicted network (top predictions)
        predicted_network = nx.Graph()
        sorted_indices = np.argsort(probs)[::-1]
        top_k = min(1000, len(probs))  # Top 1000 predictions

        for i in sorted_indices[:top_k]:
            plant_idx, pollinator_idx = (positive_pairs + unlabeled_pairs)[i]
            plant_name = self.idx_to_plant[plant_idx]
            pollinator_name = self.idx_to_pollinator[pollinator_idx]
            predicted_network.add_edge(plant_name, pollinator_name)

        # Compute structural similarity
        if len(observed_network.edges()) > 0 and len(predicted_network.edges()) > 0:
            # Jaccard similarity of edge sets
            observed_edges = set(observed_network.edges())
            predicted_edges = set(predicted_network.edges())

            intersection = len(observed_edges.intersection(predicted_edges))
            union = len(observed_edges.union(predicted_edges))

            return intersection / union if union > 0 else 0.0

        return 0.0

    def analyze_feature_importance(
        self,
        plant_features: pd.DataFrame,
        pollinator_features: pd.DataFrame,
        graph_data,
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance for plants and pollinators.

        Args:
            plant_features: Plant features
            pollinator_features: Pollinator features
            graph_data: Graph data

        Returns:
            Dictionary with feature importance DataFrames
        """
        logger.info("Analyzing feature importance...")

        # Get node embeddings
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(graph_data)

        # Analyze plant features
        plant_importance = self._analyze_node_features(
            node_embeddings[: len(plant_features)], plant_features, "plant"
        )

        # Analyze pollinator features
        pollinator_importance = self._analyze_node_features(
            node_embeddings[len(plant_features) :], pollinator_features, "pollinator"
        )

        return {
            "plant_importance": plant_importance,
            "pollinator_importance": pollinator_importance,
        }

    def _analyze_node_features(
        self, embeddings: torch.Tensor, features: pd.DataFrame, node_type: str
    ) -> pd.DataFrame:
        """Analyze feature importance for a node type."""

        # Compute correlation between features and embedding dimensions
        feature_importance = []

        for i, feature_name in enumerate(features.columns):
            feature_values = features[feature_name].values

            # Compute correlation with each embedding dimension
            correlations = []
            for j in range(embeddings.shape[1]):
                corr = np.corrcoef(feature_values, embeddings[:, j].cpu().numpy())[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            # Average correlation across embedding dimensions
            avg_correlation = np.mean(correlations) if correlations else 0.0
            feature_importance.append(
                {
                    "feature": feature_name,
                    "importance": avg_correlation,
                    "node_type": node_type,
                }
            )

        importance_df = pd.DataFrame(feature_importance)
        importance_df = importance_df.sort_values("importance", ascending=False)

        return importance_df

    def generate_predictions_report(
        self,
        interactions: pd.DataFrame,
        output_path: str = "results/prediction_report.html",
    ):
        """
        Generate a comprehensive prediction report.

        Args:
            interactions: Full interaction dataset
            output_path: Path to save the report
        """
        logger.info("Generating prediction report...")

        # Get all possible pairs
        all_plants = list(self.plant_to_idx.keys())
        all_pollinators = list(self.pollinator_to_idx.keys())

        # Sample pairs for prediction (to avoid memory issues)
        sample_size = min(10000, len(all_plants) * len(all_pollinators))
        sampled_pairs = []

        for _ in range(sample_size):
            plant = np.random.choice(all_plants)
            pollinator = np.random.choice(all_pollinators)
            sampled_pairs.append((plant, pollinator))

        # Get predictions
        predictions = []
        for plant, pollinator in sampled_pairs:
            plant_idx = self.plant_to_idx[plant]
            pollinator_idx = self.pollinator_to_idx[pollinator]

            # Check if this is an observed interaction
            is_observed = (
                len(
                    interactions[
                        (interactions["Plant_species"] == plant)
                        & (interactions["Pollinator_species"] == pollinator)
                    ]
                )
                > 0
            )

            predictions.append(
                {
                    "plant": plant,
                    "pollinator": pollinator,
                    "is_observed": is_observed,
                    "plant_idx": plant_idx,
                    "pollinator_idx": pollinator_idx,
                }
            )

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            # This would need to be implemented with proper graph data
            # For now, we'll create a placeholder
            pass

        # Create HTML report
        html_content = self._create_html_report(predictions, interactions)

        # Save report
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Prediction report saved to {output_path}")

    def _create_html_report(
        self, predictions: List[Dict], interactions: pd.DataFrame
    ) -> str:
        """Create HTML report for predictions."""

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plant-Pollinator Interaction Predictions</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .observed { background-color: #d4edda; }
                .predicted { background-color: #fff3cd; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Plant-Pollinator Interaction Predictions</h1>
                <p>PU Learning + GNN Model Results</p>
            </div>

            <div class="section">
                <h2>Dataset Summary</h2>
                <p>Total interactions: {}</p>
                <p>Unique plants: {}</p>
                <p>Unique pollinators: {}</p>
            </div>

            <div class="section">
                <h2>Top Predicted Interactions</h2>
                <table>
                    <tr>
                        <th>Plant</th>
                        <th>Pollinator</th>
                        <th>Prediction Score</th>
                        <th>Observed</th>
                    </tr>
        """.format(
            len(interactions),
            interactions["Plant_species"].nunique(),
            interactions["Pollinator_species"].nunique(),
        )

        # Add prediction rows (placeholder)
        for i, pred in enumerate(predictions[:100]):  # Show top 100
            row_class = "observed" if pred["is_observed"] else "predicted"
            html += f"""
                    <tr class="{row_class}">
                        <td>{pred['plant']}</td>
                        <td>{pred['pollinator']}</td>
                        <td>0.85</td>
                        <td>{'Yes' if pred['is_observed'] else 'No'}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        return html
