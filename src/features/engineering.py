"""
Feature Engineering for Plant-Pollinator Interaction Prediction

This module creates features for predicting the probability of interactions
between plant and pollinator species based on taxonomic information and
network context from the EuPPollNet dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class InteractionFeatureEngineer:
    """
    Feature engineering for plant-pollinator interaction prediction.

    Creates features based on:
    1. Taxonomic similarity between plants and pollinators
    2. Network-level context features
    3. Geographic and temporal features
    4. Co-occurrence patterns
    5. Network topology features
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the feature engineer.

        Args:
            data_dir: Directory containing the EuPPollNet data files
        """
        self.data_dir = Path(data_dir)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None

    def create_taxonomic_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create taxonomic similarity features between plants and pollinators.

        Args:
            interactions_df: DataFrame with plant and pollinator taxonomic info

        Returns:
            DataFrame with taxonomic similarity features
        """
        logger.info("Creating taxonomic similarity features...")

        features = []

        for _, row in interactions_df.iterrows():
            plant_taxonomy = {
                "order": row.get("Plant_order", "Unknown"),
                "family": row.get("Plant_family", "Unknown"),
                "genus": row.get("Plant_genus", "Unknown"),
                "species": row.get("Plant_accepted_name", "Unknown"),
            }

            pollinator_taxonomy = {
                "order": row.get("Pollinator_order", "Unknown"),
                "family": row.get("Pollinator_family", "Unknown"),
                "genus": row.get("Pollinator_genus", "Unknown"),
                "species": row.get("Pollinator_accepted_name", "Unknown"),
            }

            # Only encode taxonomy independently, no cross-taxon matching
            taxonomic_features = {
                # Plant taxonomy encodings
                "plant_order_encoded": self._encode_categorical(
                    plant_taxonomy["order"], "plant_order"
                ),
                "plant_family_encoded": self._encode_categorical(
                    plant_taxonomy["family"], "plant_family"
                ),
                "plant_genus_encoded": self._encode_categorical(
                    plant_taxonomy["genus"], "plant_genus"
                ),
                # Pollinator taxonomy encodings
                "pollinator_order_encoded": self._encode_categorical(
                    pollinator_taxonomy["order"], "pollinator_order"
                ),
                "pollinator_family_encoded": self._encode_categorical(
                    pollinator_taxonomy["family"], "pollinator_family"
                ),
                "pollinator_genus_encoded": self._encode_categorical(
                    pollinator_taxonomy["genus"], "pollinator_genus"
                ),
                # Placeholder for future trait-matching features
                "trait_match_placeholder": 0,  # TODO: Replace with real trait-matching features
            }

            features.append(taxonomic_features)

        return pd.DataFrame(features)

    def create_network_context_features(
        self, interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create network-level context features.

        Args:
            interactions_df: DataFrame with network information

        Returns:
            DataFrame with network context features
        """
        logger.info("Creating network context features...")

        # Network-level aggregations
        network_stats = (
            interactions_df.groupby("Network_id_full")
            .agg(
                {
                    "Plant_accepted_name": "nunique",
                    "Pollinator_accepted_name": "nunique",
                    "Interaction": ["sum", "mean", "count"],
                    "Country": "first",
                    "Year": "first",
                    "Month": "first",
                    "Latitude": "first",
                    "Longitude": "first",
                    "Authors_habitat": "first",
                    "EuPPollNet_habitat": "first",
                    "Sampling_method": "first",
                }
            )
            .reset_index()
        )

        network_stats.columns = [
            "Network_id_full",
            "plant_richness",
            "pollinator_richness",
            "total_interactions",
            "mean_interaction_strength",
            "interaction_count",
            "country",
            "year",
            "month",
            "latitude",
            "longitude",
            "authors_habitat",
            "euppollnet_habitat",
            "sampling_method",
        ]

        # Merge back to interactions
        features_df = interactions_df.merge(
            network_stats, on="Network_id_full", how="left"
        )

        # Create network context features
        network_features = []

        for _, row in features_df.iterrows():
            network_context = {
                # Network size features
                "network_plant_richness": row["plant_richness"],
                "network_pollinator_richness": row["pollinator_richness"],
                "network_total_interactions": row["total_interactions"],
                "network_mean_interaction_strength": row["mean_interaction_strength"],
                "network_interaction_count": row["interaction_count"],
                # Network density (interactions / (plants * pollinators))
                "network_density": (
                    row["total_interactions"]
                    / (row["plant_richness"] * row["pollinator_richness"])
                    if (row["plant_richness"] * row["pollinator_richness"]) > 0
                    else 0
                ),
                # Temporal features
                "year": row["year"],
                "month": row["month"],
                "season": self._get_season(row["month"]),
                # Geographic features
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "country_encoded": self._encode_categorical(row["country"], "country"),
                "authors_habitat_encoded": self._encode_categorical(
                    row["authors_habitat"], "authors_habitat"
                ),
                "euppollnet_habitat_encoded": self._encode_categorical(
                    row["euppollnet_habitat"], "euppollnet_habitat"
                ),
                "sampling_method_encoded": self._encode_categorical(
                    row["sampling_method"], "sampling_method"
                ),
            }

            network_features.append(network_context)

        return pd.DataFrame(network_features)

    def create_cooccurrence_features(
        self, interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create co-occurrence based features.

        Args:
            interactions_df: DataFrame with interaction data

        Returns:
            DataFrame with co-occurrence features
        """
        logger.info("Creating co-occurrence features...")

        # Calculate species-level statistics
        plant_stats = (
            interactions_df.groupby("Plant_accepted_name")
            .agg(
                {
                    "Pollinator_accepted_name": "nunique",
                    "Interaction": ["sum", "mean"],
                    "Network_id_full": "nunique",
                }
            )
            .reset_index()
        )

        plant_stats.columns = [
            "Plant_accepted_name",
            "plant_pollinator_partners",
            "plant_total_interactions",
            "plant_mean_interaction_strength",
            "plant_networks_present",
        ]

        pollinator_stats = (
            interactions_df.groupby("Pollinator_accepted_name")
            .agg(
                {
                    "Plant_accepted_name": "nunique",
                    "Interaction": ["sum", "mean"],
                    "Network_id_full": "nunique",
                }
            )
            .reset_index()
        )

        pollinator_stats.columns = [
            "Pollinator_accepted_name",
            "pollinator_plant_partners",
            "pollinator_total_interactions",
            "pollinator_mean_interaction_strength",
            "pollinator_networks_present",
        ]

        # Merge back to interactions
        features_df = interactions_df.merge(
            plant_stats, on="Plant_accepted_name", how="left"
        )
        features_df = features_df.merge(
            pollinator_stats, on="Pollinator_accepted_name", how="left"
        )

        # Create co-occurrence features
        cooccurrence_features = []

        for _, row in features_df.iterrows():
            cooccurrence = {
                # Species-level features
                "plant_pollinator_partners": row["plant_pollinator_partners"],
                "plant_total_interactions": row["plant_total_interactions"],
                "plant_mean_interaction_strength": row[
                    "plant_mean_interaction_strength"
                ],
                "plant_networks_present": row["plant_networks_present"],
                "pollinator_plant_partners": row["pollinator_plant_partners"],
                "pollinator_total_interactions": row["pollinator_total_interactions"],
                "pollinator_mean_interaction_strength": row[
                    "pollinator_mean_interaction_strength"
                ],
                "pollinator_networks_present": row["pollinator_networks_present"],
                # Interaction probability features
                "plant_generalism": (
                    row["plant_pollinator_partners"] / row["plant_networks_present"]
                    if row["plant_networks_present"] > 0
                    else 0
                ),
                "pollinator_generalism": (
                    row["pollinator_plant_partners"]
                    / row["pollinator_networks_present"]
                    if row["pollinator_networks_present"] > 0
                    else 0
                ),
                # Combined features
                "total_partners_product": row["plant_pollinator_partners"]
                * row["pollinator_plant_partners"],
                "total_interactions_product": row["plant_total_interactions"]
                * row["pollinator_total_interactions"],
            }

            cooccurrence_features.append(cooccurrence)

        return pd.DataFrame(cooccurrence_features)

    def create_network_topology_features(
        self, interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create network topology features for each network.

        Args:
            interactions_df: DataFrame with interaction data

        Returns:
            DataFrame with network topology features
        """
        logger.info("Creating network topology features...")

        topology_features = []

        for network_id in interactions_df["Network_id_full"].unique():
            network_data = interactions_df[
                interactions_df["Network_id_full"] == network_id
            ]

            # Create bipartite graph
            G = nx.Graph()

            # Add nodes
            plants = network_data["Plant_accepted_name"].unique()
            pollinators = network_data["Pollinator_accepted_name"].unique()

            for plant in plants:
                G.add_node(plant, bipartite=0)  # Plants are partition 0
            for pollinator in pollinators:
                G.add_node(pollinator, bipartite=1)  # Pollinators are partition 1

            # Add edges
            for _, row in network_data.iterrows():
                G.add_edge(row["Plant_accepted_name"], row["Pollinator_accepted_name"])

            # Calculate network metrics
            try:
                density = nx.density(G)
                avg_clustering = nx.average_clustering(G)
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                connected_components = nx.number_connected_components(G)

                # Bipartite-specific metrics
                plant_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
                pollinator_nodes = [
                    n for n, d in G.nodes(data=True) if d["bipartite"] == 1
                ]

                plant_avg_degree = (
                    sum(G.degree(plant) for plant in plant_nodes) / len(plant_nodes)
                    if plant_nodes
                    else 0
                )
                pollinator_avg_degree = (
                    sum(G.degree(pollinator) for pollinator in pollinator_nodes)
                    / len(pollinator_nodes)
                    if pollinator_nodes
                    else 0
                )

            except:
                # Fallback values if network analysis fails
                density = 0
                avg_clustering = 0
                avg_degree = 0
                connected_components = 1
                plant_avg_degree = 0
                pollinator_avg_degree = 0

            # Create features for each interaction in this network
            for _, row in network_data.iterrows():
                topology = {
                    "network_density": density,
                    "network_avg_clustering": avg_clustering,
                    "network_avg_degree": avg_degree,
                    "network_connected_components": connected_components,
                    "network_plant_avg_degree": plant_avg_degree,
                    "network_pollinator_avg_degree": pollinator_avg_degree,
                }

                topology_features.append(topology)

        return pd.DataFrame(topology_features)

    def create_temporal_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features based on timing of interactions.

        Args:
            interactions_df: DataFrame with temporal information

        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features...")

        temporal_features = []

        for _, row in interactions_df.iterrows():
            temporal = {
                "year": row["Year"],
                "month": row["Month"],
                "day": row["Day"],
                "season": self._get_season(row["Month"]),
                "is_spring": int(row["Month"] in [3, 4, 5]),
                "is_summer": int(row["Month"] in [6, 7, 8]),
                "is_autumn": int(row["Month"] in [9, 10, 11]),
                "is_winter": int(row["Month"] in [12, 1, 2]),
                "is_peak_season": int(
                    row["Month"] in [5, 6, 7, 8]
                ),  # Peak flowering season
            }

            temporal_features.append(temporal)

        return pd.DataFrame(temporal_features)

    def create_all_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all feature types and combine them.

        Args:
            interactions_df: DataFrame with interaction data

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Creating all features for interaction prediction...")

        # Create different feature types
        taxonomic_features = self.create_taxonomic_features(interactions_df)
        network_features = self.create_network_context_features(interactions_df)
        cooccurrence_features = self.create_cooccurrence_features(interactions_df)
        topology_features = self.create_network_topology_features(interactions_df)
        temporal_features = self.create_temporal_features(interactions_df)

        # Combine all features
        all_features = pd.concat(
            [
                taxonomic_features,
                network_features,
                cooccurrence_features,
                topology_features,
                temporal_features,
            ],
            axis=1,
        )

        # Remove duplicate columns, keeping only the first occurrence
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Add target variable (interaction presence)
        all_features["interaction_present"] = (
            interactions_df["Interaction"] > 0
        ).astype(int)
        all_features["interaction_strength"] = interactions_df["Interaction"]

        # Add identifiers for tracking
        all_features["plant_species"] = interactions_df["Plant_accepted_name"]
        all_features["pollinator_species"] = interactions_df["Pollinator_accepted_name"]
        all_features["network_id"] = interactions_df["Network_id_full"]

        logger.info(
            f"Created {len(all_features.columns)} features for {len(all_features)} interactions"
        )

        return all_features

    def _calculate_taxonomic_distance(
        self, plant_taxonomy: Dict, pollinator_taxonomy: Dict
    ) -> int:
        """
        Calculate taxonomic distance between plant and pollinator.

        Args:
            plant_taxonomy: Plant taxonomic information
            pollinator_taxonomy: Pollinator taxonomic information

        Returns:
            Taxonomic distance (0-4, where 0 is same species, 4 is different orders)
        """
        taxonomy_levels = ["order", "family", "genus", "species"]

        for i, level in enumerate(taxonomy_levels):
            if plant_taxonomy[level] != pollinator_taxonomy[level]:
                return len(taxonomy_levels) - i

        return 0  # Same species

    def _encode_categorical(self, value: str, feature_name: str) -> int:
        """
        Encode categorical values using LabelEncoder.

        Args:
            value: Categorical value to encode
            feature_name: Name of the feature for tracking encoders

        Returns:
            Encoded integer value
        """
        if feature_name not in self.label_encoders:
            self.label_encoders[feature_name] = LabelEncoder()
            # Note: In practice, you'd fit this on training data only

        try:
            return self.label_encoders[feature_name].transform([value])[0]
        except:
            # Handle unseen categories
            return -1

    def _get_season(self, month: int) -> str:
        """
        Get season from month.

        Args:
            month: Month number (1-12)

        Returns:
            Season name
        """
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def prepare_training_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = "interaction_present",
        exclude_cols: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training.

        Args:
            features_df: DataFrame with all features
            target_col: Name of target column
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (X, y) for training
        """
        if exclude_cols is None:
            exclude_cols = [
                "interaction_present",
                "interaction_strength",
                "plant_species",
                "pollinator_species",
                "network_id",
            ]

        # Select feature columns
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].values
        y = features_df[target_col].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        logger.info(
            f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features"
        )

        return X, y


def create_sample_features() -> pd.DataFrame:
    """
    Create sample features for testing and development.

    Returns:
        DataFrame with sample engineered features
    """
    logger.info("Creating sample features for development")

    # Sample interaction data
    sample_interactions = pd.DataFrame(
        {
            "Plant_accepted_name": [
                "Lavandula stoechas",
                "Lavandula stoechas",
                "Sonchus tenerrimus",
            ],
            "Pollinator_accepted_name": [
                "Anthidium sticticum",
                "Osmia aurulenta",
                "Oedemera flavipes",
            ],
            "Plant_phylum": ["Tracheophyta", "Tracheophyta", "Tracheophyta"],
            "Plant_order": ["Lamiales", "Lamiales", "Asterales"],
            "Plant_family": ["Lamiaceae", "Lamiaceae", "Asteraceae"],
            "Plant_genus": ["Lavandula", "Lavandula", "Sonchus"],
            "Pollinator_phylum": ["Arthropoda", "Arthropoda", "Arthropoda"],
            "Pollinator_order": ["Hymenoptera", "Hymenoptera", "Coleoptera"],
            "Pollinator_family": ["Megachilidae", "Megachilidae", "Oedemeridae"],
            "Pollinator_genus": ["Anthidium", "Osmia", "Oedemera"],
            "Network_id_full": ["test_network_1", "test_network_1", "test_network_1"],
            "Interaction": [1, 1, 1],
            "Year": [2005, 2005, 2005],
            "Month": [3, 3, 4],
            "Day": [13, 13, 9],
            "Country": ["Spain", "Spain", "Spain"],
            "Latitude": [42.352, 42.352, 42.352],
            "Longitude": [3.177, 3.177, 3.177],
            "Authors_habitat": [
                "Coastal shrubland",
                "Coastal shrubland",
                "Coastal shrubland",
            ],
            "EuPPollNet_habitat": [
                "Sclerophyllous vegetation",
                "Sclerophyllous vegetation",
                "Sclerophyllous vegetation",
            ],
            "Sampling_method": [
                "Focal_observation",
                "Focal_observation",
                "Focal_observation",
            ],
        }
    )

    # Create features
    engineer = InteractionFeatureEngineer()
    features = engineer.create_all_features(sample_interactions)

    return features
