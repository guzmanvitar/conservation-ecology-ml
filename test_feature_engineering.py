#!/usr/bin/env python3
"""
Test script for feature engineering module.

This script demonstrates the feature engineering capabilities for
predicting plant-pollinator interactions based on taxonomic information.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append("src")

import logging

import pandas as pd

from data.loader import EuPPollNetLoader
from features.engineering import InteractionFeatureEngineer, create_sample_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_engineering():
    """Test the feature engineering pipeline."""

    print("=" * 80)
    print("FEATURE ENGINEERING TEST")
    print("=" * 80)

    # First, let's test with sample data
    print("\n1. Testing with sample data...")
    sample_features = create_sample_features()
    print(f"Sample features shape: {sample_features.shape}")
    print(f"Sample features columns: {list(sample_features.columns)}")

    # Show some sample features
    print("\nSample feature values:")
    print(sample_features.head(3).to_string())

    # Now let's test with real data (if available)
    print("\n2. Testing with real EuPPollNet data...")

    try:
        # Load data
        loader = EuPPollNetLoader("data/raw")
        data = loader.load_all_data()
        interactions = data["interactions"]

        print(f"Loaded {len(interactions):,} interactions")
        print(f"Networks: {interactions['Network_id_full'].nunique():,}")
        print(f"Plant species: {interactions['Plant_accepted_name'].nunique():,}")
        print(
            f"Pollinator species: {interactions['Pollinator_accepted_name'].nunique():,}"
        )

        # Create feature engineer
        engineer = InteractionFeatureEngineer("data/raw")

        # Create features for a subset of data (for speed)
        print("\nCreating features for a sample of interactions...")
        sample_interactions = interactions.sample(
            n=min(1000, len(interactions)), random_state=42
        )

        # Create all features
        features = engineer.create_all_features(sample_interactions)

        print(f"\nFeature engineering completed!")
        print(f"Features shape: {features.shape}")
        print(f"Number of features: {len(features.columns)}")

        # Show feature categories
        print("\nFeature categories:")
        feature_categories = {
            "Taxonomic": [
                col
                for col in features.columns
                if any(
                    x in col
                    for x in ["phylum", "order", "family", "genus", "taxonomic"]
                )
            ],
            "Network Context": [col for col in features.columns if "network_" in col],
            "Co-occurrence": [
                col
                for col in features.columns
                if any(x in col for x in ["partners", "generalism", "product"])
            ],
            "Temporal": [
                col
                for col in features.columns
                if any(x in col for x in ["year", "month", "season", "is_"])
            ],
            "Target": [col for col in features.columns if "interaction_" in col],
            "Identifiers": [
                col
                for col in features.columns
                if any(x in col for x in ["species", "network_id"])
            ],
        }

        for category, cols in feature_categories.items():
            if cols:
                print(
                    f"  {category} ({len(cols)} features): {cols[:5]}{'...' if len(cols) > 5 else ''}"
                )

        # Show feature statistics
        print("\nFeature statistics:")
        numeric_features = features.select_dtypes(include=[np.number])
        print(f"  Numeric features: {len(numeric_features.columns)}")
        print(
            f"  Categorical features: {len(features.columns) - len(numeric_features.columns)}"
        )

        # Show target distribution
        if "interaction_present" in features.columns:
            target_dist = features["interaction_present"].value_counts()
            print(f"\nTarget distribution:")
            print(
                f"  Interactions present: {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(features)*100:.1f}%)"
            )
            print(
                f"  No interactions: {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(features)*100:.1f}%)"
            )

        # Prepare training data
        print("\nPreparing training data...")
        X, y = engineer.prepare_training_data(features)
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Feature matrix sparsity: {(X == 0).sum() / X.size * 100:.1f}%")

        # Show some example predictions
        print("\nExample feature values for first 3 interactions:")
        example_features = features.head(3)
        for i, (_, row) in enumerate(example_features.iterrows()):
            plant_species = (
                str(row["plant_species"])
                if pd.notna(row["plant_species"])
                else "Unknown"
            )
            pollinator_species = (
                str(row["pollinator_species"])
                if pd.notna(row["pollinator_species"])
                else "Unknown"
            )
            print(f"\nInteraction {i+1}: {plant_species} <-> {pollinator_species}")

            # Show available features
            plant_order = (
                row["plant_order_encoded"]
                if pd.notna(row["plant_order_encoded"])
                else "NA"
            )
            pollinator_order = (
                row["pollinator_order_encoded"]
                if pd.notna(row["pollinator_order_encoded"])
                else "NA"
            )
            trait_placeholder = (
                row["trait_match_placeholder"]
                if pd.notna(row["trait_match_placeholder"])
                else "NA"
            )
            net_density = (
                f"{row['network_density']:.4f}"
                if pd.notna(row["network_density"])
                else "NA"
            )
            plant_gen = (
                f"{row['plant_generalism']:.2f}"
                if pd.notna(row["plant_generalism"])
                else "NA"
            )
            poll_gen = (
                f"{row['pollinator_generalism']:.2f}"
                if pd.notna(row["pollinator_generalism"])
                else "NA"
            )

            if pd.isna(row["interaction_present"]):
                interaction_present = "NA"
            else:
                try:
                    interaction_present = int(row["interaction_present"])
                except Exception:
                    interaction_present = row["interaction_present"]

            print(f"  Plant order encoded: {plant_order}")
            print(f"  Pollinator order encoded: {pollinator_order}")
            print(f"  Trait match placeholder: {trait_placeholder}")
            print(f"  Network density: {net_density}")
            print(f"  Plant generalism: {plant_gen}")
            print(f"  Pollinator generalism: {poll_gen}")
            print(f"  Interaction present: {interaction_present}")

        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 80)
        print(f"✅ Successfully created {len(features.columns)} features")
        print(
            f"✅ Features cover taxonomic, network, temporal, and co-occurrence patterns"
        )
        print(f"✅ Ready for machine learning model training")
        print(f"✅ Target variable: interaction_present (binary classification)")

        return features, X, y

    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        print("Using sample data only...")
        return sample_features, None, None


if __name__ == "__main__":
    import numpy as np

    features, X, y = test_feature_engineering()
