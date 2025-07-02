"""
Feature engineering for PU Learning with plant-pollinator interactions.

This module creates features for plants and pollinators that are suitable
for PU learning in ecological networks.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class PUFeatureEngineer:
    """
    Feature engineer for PU Learning with plant-pollinator networks.

    Creates features that help distinguish between positive and unlabeled
    plant-pollinator pairs.
    """

    def __init__(
        self,
        use_traits: bool = True,
        use_network_features: bool = True,
        use_temporal_features: bool = True,
        use_spatial_features: bool = True,
    ):
        """
        Initialize feature engineer.

        Args:
            use_traits: Whether to use species trait features
            use_network_features: Whether to use network topology features
            use_temporal_features: Whether to use temporal features
            use_spatial_features: Whether to use spatial features
        """
        self.use_traits = use_traits
        self.use_network_features = use_network_features
        self.use_temporal_features = use_temporal_features
        self.use_spatial_features = use_spatial_features

        # Scalers and encoders
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def create_plant_features(
        self,
        plant_taxonomy: pd.DataFrame,
        interactions: pd.DataFrame,
        flower_counts: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create features for plant species.

        Args:
            plant_taxonomy: Plant taxonomic data
            interactions: Interaction data
            flower_counts: Optional flower abundance data

        Returns:
            DataFrame with plant features
        """
        logger.info("Creating plant features...")

        # Start with basic taxonomic features
        plant_features = plant_taxonomy.copy()

        # Encode categorical features
        categorical_cols = ["Family", "Genus", "Order"]
        for col in categorical_cols:
            if col in plant_features.columns:
                le = LabelEncoder()
                plant_features[f"{col}_encoded"] = le.fit_transform(
                    plant_features[col].fillna("Unknown")
                )
                self.label_encoders[f"plant_{col}"] = le

        # Create interaction-based features
        if self.use_network_features:
            plant_features = self._add_network_features_plants(
                plant_features, interactions
            )

        # Add temporal features
        if self.use_temporal_features:
            plant_features = self._add_temporal_features_plants(
                plant_features, interactions
            )

        # Add spatial features
        if self.use_spatial_features:
            plant_features = self._add_spatial_features_plants(
                plant_features, interactions
            )

        # Add flower abundance features if available
        if flower_counts is not None:
            plant_features = self._add_flower_features(plant_features, flower_counts)

        # Create trait-based features
        if self.use_traits:
            plant_features = self._add_plant_traits(plant_features)

        # Remove original categorical columns
        for col in categorical_cols:
            if col in plant_features.columns:
                plant_features = plant_features.drop(columns=[col])

        # Handle missing values
        plant_features = plant_features.fillna(0)

        # Set index to Accepted_name for downstream compatibility
        if "Accepted_name" in plant_features.columns:
            plant_features = plant_features.drop_duplicates(subset="Accepted_name")
            plant_features = plant_features.set_index("Accepted_name")

        logger.info(
            f"Created {plant_features.shape[1]} features for {len(plant_features)} plants"
        )

        return plant_features

    def create_pollinator_features(
        self, pollinator_taxonomy: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features for pollinator species.

        Args:
            pollinator_taxonomy: Pollinator taxonomic data
            interactions: Interaction data

        Returns:
            DataFrame with pollinator features
        """
        logger.info("Creating pollinator features...")

        # Start with basic taxonomic features
        pollinator_features = pollinator_taxonomy.copy()

        # Encode categorical features
        categorical_cols = ["Family", "Genus", "Order"]
        for col in categorical_cols:
            if col in pollinator_features.columns:
                le = LabelEncoder()
                pollinator_features[f"{col}_encoded"] = le.fit_transform(
                    pollinator_features[col].fillna("Unknown")
                )
                self.label_encoders[f"pollinator_{col}"] = le

        # Create interaction-based features
        if self.use_network_features:
            pollinator_features = self._add_network_features_pollinators(
                pollinator_features, interactions
            )

        # Add temporal features
        if self.use_temporal_features:
            pollinator_features = self._add_temporal_features_pollinators(
                pollinator_features, interactions
            )

        # Add spatial features
        if self.use_spatial_features:
            pollinator_features = self._add_spatial_features_pollinators(
                pollinator_features, interactions
            )

        # Create trait-based features
        if self.use_traits:
            pollinator_features = self._add_pollinator_traits(pollinator_features)

        # Remove original categorical columns
        for col in categorical_cols:
            if col in pollinator_features.columns:
                pollinator_features = pollinator_features.drop(columns=[col])

        # Handle missing values
        pollinator_features = pollinator_features.fillna(0)

        # Set index to Accepted_name for downstream compatibility
        if "Accepted_name" in pollinator_features.columns:
            pollinator_features = pollinator_features.drop_duplicates(
                subset="Accepted_name"
            )
            pollinator_features = pollinator_features.set_index("Accepted_name")

        logger.info(
            f"Created {pollinator_features.shape[1]} features for {len(pollinator_features)} pollinators"
        )

        return pollinator_features

    def _add_network_features_plants(
        self, plant_features: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add network topology features for plants."""

        # Calculate degree (number of pollinator partners)
        plant_degree = interactions.groupby("Plant_accepted_name")[
            "Pollinator_accepted_name"
        ].nunique()
        plant_features["degree"] = plant_features.index.map(plant_degree).fillna(0)

        # Calculate interaction frequency
        plant_freq = interactions.groupby("Plant_accepted_name").size()
        plant_features["interaction_frequency"] = plant_features.index.map(
            plant_freq
        ).fillna(0)

        # Calculate specialization (inverse of degree)
        plant_features["specialization"] = 1 / (plant_features["degree"] + 1)

        # Calculate pollinator diversity (Shannon index)
        def shannon_diversity(group):
            if len(group) == 0:
                return 0
            counts = group.value_counts()
            probs = counts / counts.sum()
            return -np.sum(probs * np.log(probs + 1e-8))

        plant_diversity = interactions.groupby("Plant_accepted_name")[
            "Pollinator_accepted_name"
        ].apply(shannon_diversity)
        plant_features["pollinator_diversity"] = plant_features.index.map(
            plant_diversity
        ).fillna(0)

        return plant_features

    def _add_network_features_pollinators(
        self, pollinator_features: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add network topology features for pollinators."""

        # Calculate degree (number of plant partners)
        pollinator_degree = interactions.groupby("Pollinator_accepted_name")[
            "Plant_accepted_name"
        ].nunique()
        pollinator_features["degree"] = pollinator_features.index.map(
            pollinator_degree
        ).fillna(0)

        # Calculate interaction frequency
        pollinator_freq = interactions.groupby("Pollinator_accepted_name").size()
        pollinator_features["interaction_frequency"] = pollinator_features.index.map(
            pollinator_freq
        ).fillna(0)

        # Calculate specialization (inverse of degree)
        pollinator_features["specialization"] = 1 / (pollinator_features["degree"] + 1)

        # Calculate plant diversity (Shannon index)
        def shannon_diversity(group):
            if len(group) == 0:
                return 0
            counts = group.value_counts()
            probs = counts / counts.sum()
            return -np.sum(probs * np.log(probs + 1e-8))

        pollinator_diversity = interactions.groupby("Pollinator_accepted_name")[
            "Plant_accepted_name"
        ].apply(shannon_diversity)
        pollinator_features["plant_diversity"] = pollinator_features.index.map(
            pollinator_diversity
        ).fillna(0)

        return pollinator_features

    def _add_temporal_features_plants(
        self, plant_features: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add temporal features for plants."""

        # Flowering season length
        plant_season = interactions.groupby("Plant_accepted_name").agg(
            {"Month": ["min", "max", "nunique"]}
        )
        plant_features["flowering_start"] = plant_features.index.map(
            plant_season[("Month", "min")]
        ).fillna(0)
        plant_features["flowering_end"] = plant_features.index.map(
            plant_season[("Month", "max")]
        ).fillna(0)
        plant_features["flowering_duration"] = plant_features.index.map(
            plant_season[("Month", "nunique")]
        ).fillna(0)

        # Temporal specialization (interactions spread across months)
        plant_features["temporal_specialization"] = (
            plant_features["flowering_duration"] / 12.0
        )

        return plant_features

    def _add_temporal_features_pollinators(
        self, pollinator_features: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add temporal features for pollinators."""

        # Activity season length
        pollinator_season = interactions.groupby("Pollinator_accepted_name").agg(
            {"Month": ["min", "max", "nunique"]}
        )
        pollinator_features["activity_start"] = pollinator_features.index.map(
            pollinator_season[("Month", "min")]
        ).fillna(0)
        pollinator_features["activity_end"] = pollinator_features.index.map(
            pollinator_season[("Month", "max")]
        ).fillna(0)
        pollinator_features["activity_duration"] = pollinator_features.index.map(
            pollinator_season[("Month", "nunique")]
        ).fillna(0)

        # Temporal specialization
        pollinator_features["temporal_specialization"] = (
            pollinator_features["activity_duration"] / 12.0
        )

        return pollinator_features

    def _add_spatial_features_plants(
        self, plant_features: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add spatial features for plants."""

        # Geographic range (number of countries)
        plant_countries = interactions.groupby("Plant_accepted_name")[
            "Country"
        ].nunique()
        plant_features["geographic_range"] = plant_features.index.map(
            plant_countries
        ).fillna(0)

        # Number of networks
        plant_networks = interactions.groupby("Plant_accepted_name")[
            "Network_id_full"
        ].nunique()
        plant_features["network_presence"] = plant_features.index.map(
            plant_networks
        ).fillna(0)

        return plant_features

    def _add_spatial_features_pollinators(
        self, pollinator_features: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add spatial features for pollinators."""

        # Geographic range (number of countries)
        pollinator_countries = interactions.groupby("Pollinator_accepted_name")[
            "Country"
        ].nunique()
        pollinator_features["geographic_range"] = pollinator_features.index.map(
            pollinator_countries
        ).fillna(0)

        # Number of networks
        pollinator_networks = interactions.groupby("Pollinator_accepted_name")[
            "Network_id_full"
        ].nunique()
        pollinator_features["network_presence"] = pollinator_features.index.map(
            pollinator_networks
        ).fillna(0)

        return pollinator_features

    def _add_flower_features(
        self, plant_features: pd.DataFrame, flower_counts: pd.DataFrame
    ) -> pd.DataFrame:
        """Add flower abundance features."""

        # Average flower abundance
        flower_avg = flower_counts.groupby("Plant_species")["Flower_count"].mean()
        plant_features["avg_flower_abundance"] = plant_features.index.map(
            flower_avg
        ).fillna(0)

        # Flower abundance variability
        flower_std = flower_counts.groupby("Plant_species")["Flower_count"].std()
        plant_features["flower_abundance_std"] = plant_features.index.map(
            flower_std
        ).fillna(0)

        return plant_features

    def _add_plant_traits(self, plant_features: pd.DataFrame) -> pd.DataFrame:
        """Add plant trait features."""

        # Create dummy features for common plant families
        common_families = [
            "Asteraceae",
            "Fabaceae",
            "Rosaceae",
            "Lamiaceae",
            "Brassicaceae",
        ]
        for family in common_families:
            plant_features[f"family_{family.lower()}"] = (
                plant_features["Family_encoded"]
                == self.label_encoders.get("plant_Family", LabelEncoder())
                .fit([family])
                .transform([family])[0]
            ).astype(int)

        # Add phylogenetic features (simplified)
        plant_features["is_woody"] = plant_features["Family_encoded"].isin(
            [0, 1, 2]
        )  # Example
        plant_features["is_herbaceous"] = ~plant_features["is_woody"]

        return plant_features

    def _add_pollinator_traits(self, pollinator_features: pd.DataFrame) -> pd.DataFrame:
        """Add pollinator trait features."""

        # Create dummy features for common pollinator families
        common_families = [
            "Apidae",
            "Syrphidae",
            "Halictidae",
            "Andrenidae",
            "Megachilidae",
        ]
        for family in common_families:
            pollinator_features[f"family_{family.lower()}"] = (
                pollinator_features["Family_encoded"]
                == self.label_encoders.get("pollinator_Family", LabelEncoder())
                .fit([family])
                .transform([family])[0]
            ).astype(int)

        # Add functional group features
        pollinator_features["is_bee"] = pollinator_features["Family_encoded"].isin(
            [0, 1, 2, 3]
        )  # Example
        pollinator_features["is_hoverfly"] = pollinator_features["Family_encoded"].isin(
            [4]
        )  # Example
        pollinator_features["is_other"] = ~(
            pollinator_features["is_bee"] | pollinator_features["is_hoverfly"]
        )

        return pollinator_features

    def create_pair_features(
        self,
        plant_features: pd.DataFrame,
        pollinator_features: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create features for plant-pollinator pairs.

        Args:
            plant_features: Plant features
            pollinator_features: Pollinator features
            interactions: Interaction data

        Returns:
            DataFrame with pair features
        """
        logger.info("Creating pair features...")

        # Get all unique pairs
        all_plants = plant_features.index.tolist()
        all_pollinators = pollinator_features.index.tolist()

        # Create all possible pairs
        pairs = []
        for plant in all_plants:
            for pollinator in all_pollinators:
                pairs.append((plant, pollinator))

        pair_df = pd.DataFrame(
            pairs, columns=["Plant_accepted_name", "Pollinator_accepted_name"]
        )

        # Add plant features
        for col in plant_features.columns:
            if col not in ["Plant_accepted_name"]:
                pair_df[f"plant_{col}"] = pair_df["Plant_accepted_name"].map(
                    plant_features[col]
                )

        # Add pollinator features
        for col in pollinator_features.columns:
            if col not in ["Pollinator_accepted_name"]:
                pair_df[f"pollinator_{col}"] = pair_df["Pollinator_accepted_name"].map(
                    pollinator_features[col]
                )

        # Add interaction features
        pair_df["has_interaction"] = (
            pair_df.set_index(["Plant_accepted_name", "Pollinator_accepted_name"])
            .index.isin(
                interactions.set_index(
                    ["Plant_accepted_name", "Pollinator_accepted_name"]
                ).index
            )
            .astype(int)
        )

        # Add ecological compatibility features
        pair_df = self._add_ecological_compatibility_features(pair_df, interactions)

        # Handle missing values
        pair_df = pair_df.fillna(0)

        logger.info(f"Created features for {len(pair_df)} plant-pollinator pairs")

        return pair_df

    def _add_ecological_compatibility_features(
        self, pair_df: pd.DataFrame, interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add ecological compatibility features for pairs."""

        # Temporal overlap
        def temporal_overlap(plant, pollinator):
            plant_months = set(
                interactions[interactions["Plant_accepted_name"] == plant]["Month"]
            )
            pollinator_months = set(
                interactions[interactions["Pollinator_accepted_name"] == pollinator][
                    "Month"
                ]
            )
            overlap = len(plant_months.intersection(pollinator_months))
            total = len(plant_months.union(pollinator_months))
            return overlap / total if total > 0 else 0

        pair_df["temporal_overlap"] = pair_df.apply(
            lambda x: temporal_overlap(
                x["Plant_accepted_name"], x["Pollinator_accepted_name"]
            ),
            axis=1,
        )

        # Spatial overlap
        def spatial_overlap(plant, pollinator):
            plant_countries = set(
                interactions[interactions["Plant_accepted_name"] == plant]["Country"]
            )
            pollinator_countries = set(
                interactions[interactions["Pollinator_accepted_name"] == pollinator][
                    "Country"
                ]
            )
            overlap = len(plant_countries.intersection(pollinator_countries))
            total = len(plant_countries.union(pollinator_countries))
            return overlap / total if total > 0 else 0

        pair_df["spatial_overlap"] = pair_df.apply(
            lambda x: spatial_overlap(
                x["Plant_accepted_name"], x["Pollinator_accepted_name"]
            ),
            axis=1,
        )

        # Network overlap (shared partners)
        def network_overlap(plant, pollinator):
            plant_partners = set(
                interactions[interactions["Plant_accepted_name"] == plant][
                    "Pollinator_accepted_name"
                ]
            )
            pollinator_partners = set(
                interactions[interactions["Pollinator_accepted_name"] == pollinator][
                    "Plant_accepted_name"
                ]
            )

            # Jaccard similarity
            intersection = len(plant_partners.intersection(pollinator_partners))
            union = len(plant_partners.union(pollinator_partners))
            return intersection / union if union > 0 else 0

        pair_df["network_overlap"] = pair_df.apply(
            lambda x: network_overlap(
                x["Plant_accepted_name"], x["Pollinator_accepted_name"]
            ),
            axis=1,
        )

        return pair_df

    def scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using StandardScaler.

        Args:
            features: Features to scale

        Returns:
            Scaled features
        """
        # Get numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns

        # Scale features
        scaled_features = features.copy()
        scaled_features[numeric_cols] = self.scaler.fit_transform(
            features[numeric_cols]
        )

        return scaled_features

    def fit_transform(
        self,
        plant_taxonomy: pd.DataFrame,
        pollinator_taxonomy: pd.DataFrame,
        interactions: pd.DataFrame,
        flower_counts: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit the feature engineer and transform the data.

        Args:
            plant_taxonomy: Plant taxonomic data
            pollinator_taxonomy: Pollinator taxonomic data
            interactions: Interaction data
            flower_counts: Optional flower abundance data

        Returns:
            Tuple of (plant_features, pollinator_features, pair_features)
        """
        # Create features
        plant_features = self.create_plant_features(
            plant_taxonomy, interactions, flower_counts
        )
        pollinator_features = self.create_pollinator_features(
            pollinator_taxonomy, interactions
        )
        pair_features = self.create_pair_features(
            plant_features, pollinator_features, interactions
        )

        # Scale features
        plant_features = self.scale_features(plant_features)
        pollinator_features = self.scale_features(pollinator_features)

        return plant_features, pollinator_features, pair_features
