#!/usr/bin/env python3
"""
Test pipeline with small subsample of EuPPollNet data.

This script loads a small subset of the data to test the pipeline
without running into memory issues, focusing on a specific network ID.
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.loader import EuPPollNetLoader
from features.pu_features import PUFeatureEngineer
from models.pu_gnn import BipartiteGNN, PUGNNTrainer

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_default_network_id(loader: EuPPollNetLoader) -> str:
    """
    Get a default network ID for testing.

    Args:
        loader: EuPPollNetLoader instance

    Returns:
        Default network ID string
    """
    # Load interactions to find available networks
    interactions = loader.load_interactions()

    # Get network statistics
    network_counts = interactions["Network_id_full"].value_counts()

    # Find a network with reasonable size (not too small, not too large)
    # Look for networks with 50-500 interactions
    suitable_networks = network_counts[(network_counts >= 50) & (network_counts <= 500)]

    if len(suitable_networks) > 0:
        # Select the first suitable network
        default_network = suitable_networks.index[0]
        logger.info(
            f"Selected default network: {default_network} with {suitable_networks.iloc[0]} interactions"
        )
    else:
        # Fallback to the network with most interactions (but limit to 500)
        default_network = network_counts.index[0]
        logger.info(
            f"Selected fallback network: {default_network} with {network_counts.iloc[0]} interactions"
        )

    return default_network


def create_network_sample(
    loader: EuPPollNetLoader, network_id: str, max_interactions: int = 500
) -> dict:
    """
    Create a sample focused on a specific network.

    Args:
        loader: EuPPollNetLoader instance
        network_id: Network ID to focus on
        max_interactions: Maximum number of interactions to include

    Returns:
        Dictionary with network-focused sample of data
    """
    logger.info(f"Creating sample focused on network: {network_id}")

    # Load all data first
    full_data = loader.load_all_data()

    # Filter interactions to the specific network
    network_interactions = full_data["interactions"][
        full_data["interactions"]["Network_id_full"] == network_id
    ].copy()

    # If network is too large, sample from it
    if len(network_interactions) > max_interactions:
        network_interactions = network_interactions.sample(
            n=max_interactions, random_state=42
        )
        logger.info(
            f"Sampled {max_interactions} interactions from network {network_id}"
        )

    # Get unique species from the network
    unique_plants = network_interactions["Plant_accepted_name"].unique()
    unique_pollinators = network_interactions["Pollinator_accepted_name"].unique()

    # Filter taxonomy to only include species in the network
    plants_taxonomy_sample = full_data["plants_taxonomy"][
        full_data["plants_taxonomy"]["Accepted_name"].isin(unique_plants)
    ].copy()

    pollinators_taxonomy_sample = full_data["pollinators_taxonomy"][
        full_data["pollinators_taxonomy"]["Accepted_name"].isin(unique_pollinators)
    ].copy()

    # Filter metadata to only include the study
    unique_studies = network_interactions["Study_id"].unique()
    metadata_sample = full_data["metadata"][
        full_data["metadata"]["Study_id"].isin(unique_studies)
    ].copy()

    logger.info("Network sample contains:")
    logger.info(f"  - {len(network_interactions)} interactions")
    logger.info(f"  - {len(unique_plants)} plant species")
    logger.info(f"  - {len(unique_pollinators)} pollinator species")
    logger.info(f"  - {len(unique_studies)} studies")
    logger.info(f"  - Network ID: {network_id}")

    return {
        "interactions": network_interactions,
        "plants_taxonomy": plants_taxonomy_sample,
        "pollinators_taxonomy": pollinators_taxonomy_sample,
        "metadata": metadata_sample,
        "network_id": network_id,
    }


def test_feature_engineering(data: dict):
    """Test feature engineering on network sample."""
    logger.info("=== Testing Feature Engineering ===")

    feature_engineer = PUFeatureEngineer(
        use_traits=True,
        use_network_features=True,
        use_temporal_features=True,
        use_spatial_features=True,
    )

    # Create features
    plant_features = feature_engineer.create_plant_features(
        plant_taxonomy=data["plants_taxonomy"], interactions=data["interactions"]
    )

    pollinator_features = feature_engineer.create_pollinator_features(
        pollinator_taxonomy=data["pollinators_taxonomy"],
        interactions=data["interactions"],
    )

    pair_features = feature_engineer.create_pair_features(
        plant_features=plant_features,
        pollinator_features=pollinator_features,
        interactions=data["interactions"],
    )

    logger.info(f"Created {plant_features.shape[1]} plant features")
    logger.info(f"Created {pollinator_features.shape[1]} pollinator features")
    logger.info(f"Created {pair_features.shape[1]} pair features")
    logger.info(f"Total pairs: {len(pair_features)}")

    return plant_features, pollinator_features, pair_features, feature_engineer


def test_gnn_model(plant_features, pollinator_features, pair_features, data):
    """Test GNN model on network sample."""
    logger.info("=== Testing GNN Model ===")

    # Get numeric feature counts for model initialization
    plant_numeric = plant_features.select_dtypes(include=[np.number])
    pollinator_numeric = pollinator_features.select_dtypes(include=[np.number])

    plant_feature_count = plant_numeric.shape[1]
    pollinator_feature_count = pollinator_numeric.shape[1]

    # Create a simple bipartite graph
    model = BipartiteGNN(
        plant_features=plant_feature_count,
        pollinator_features=pollinator_feature_count,
        hidden_dim=64,  # Smaller for testing
        num_layers=2,
        dropout=0.3,
        conv_type="gcn",
    )

    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create trainer
    trainer = PUGNNTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-5,
        device="cpu",  # Use CPU for testing
    )

    logger.info("Trainer created successfully")

    return model, trainer


def test_model_evaluation(
    model, trainer, plant_features, pollinator_features, pair_features, data
):
    """Test model evaluation on network sample, including model performance metrics."""
    logger.info("=== Testing Model Evaluation ===")

    # Prepare graph data and pairs
    graph_data, plant_to_idx, pollinator_to_idx, num_plants = (
        trainer.create_bipartite_graph(
            data["interactions"], plant_features, pollinator_features
        )
    )

    # Use the mappings returned by create_bipartite_graph

    # Positive pairs (observed interactions)
    positive_pairs = [
        (
            plant_to_idx[row["Plant_accepted_name"]],
            pollinator_to_idx[row["Pollinator_accepted_name"]],
        )
        for _, row in data["interactions"].iterrows()
        if row["Plant_accepted_name"] in plant_to_idx
        and row["Pollinator_accepted_name"] in pollinator_to_idx
    ]

    # Unlabeled pairs (sampled)
    unlabeled_pairs = trainer.sample_unlabeled_pairs(
        data["interactions"],
        plant_to_idx,
        pollinator_to_idx,
        num_samples=len(positive_pairs),
    )

    # Shuffle and split into train/test
    import random

    random.seed(42)
    random.shuffle(positive_pairs)
    random.shuffle(unlabeled_pairs)
    split = int(0.8 * len(positive_pairs))
    train_pos, test_pos = positive_pairs[:split], positive_pairs[split:]
    train_unl, test_unl = unlabeled_pairs[:split], unlabeled_pairs[split:]

    # Train for a few epochs
    logger.info("Training model for 10 epochs on small sample...")
    trainer.train(
        graph_data,
        train_pos,
        train_unl,
        num_epochs=10,
        val_split=0.0,
        num_plants=num_plants,
    )

    # Evaluate on test set
    metrics = trainer.evaluate(graph_data, test_pos, test_unl)
    logger.info(
        f"Test AUC: {metrics['auc']:.4f}, AP: {metrics['average_precision']:.4f}, Loss: {metrics['loss']:.4f}"
    )

    # For testing, we'll create a simple evaluation summary
    evaluation_summary = {
        "network_id": data["network_id"],
        "num_interactions": len(data["interactions"]),
        "num_plant_species": len(data["interactions"]["Plant_accepted_name"].unique()),
        "num_pollinator_species": len(
            data["interactions"]["Pollinator_accepted_name"].unique()
        ),
        "num_pairs": len(pair_features),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "plant_features": plant_features.shape[1],
        "pollinator_features": pollinator_features.shape[1],
        "pair_features": pair_features.shape[1],
        "network_density": len(data["interactions"])
        / (
            len(data["interactions"]["Plant_accepted_name"].unique())
            * len(data["interactions"]["Pollinator_accepted_name"].unique())
        ),
        "avg_interaction_strength": data["interactions"]["Interaction"].mean(),
        "max_interaction_strength": data["interactions"]["Interaction"].max(),
        "min_interaction_strength": data["interactions"]["Interaction"].min(),
        # Model performance metrics:
        "test_auc": metrics["auc"],
        "test_average_precision": metrics["average_precision"],
        "test_loss": metrics["loss"],
    }

    # Add network-level statistics
    network_stats = (
        data["interactions"]
        .groupby("Network_id_full")
        .agg(
            {
                "Plant_accepted_name": "nunique",
                "Pollinator_accepted_name": "nunique",
                "Interaction": ["sum", "mean", "std"],
            }
        )
        .round(4)
    )

    # Convert to JSON-serializable format
    network_stats_dict = {}
    for network_id in network_stats.index:
        network_stats_dict[str(network_id)] = {
            "plant_species": int(
                network_stats.loc[network_id, ("Plant_accepted_name", "nunique")]
            ),
            "pollinator_species": int(
                network_stats.loc[network_id, ("Pollinator_accepted_name", "nunique")]
            ),
            "interaction_sum": float(
                network_stats.loc[network_id, ("Interaction", "sum")]
            ),
            "interaction_mean": float(
                network_stats.loc[network_id, ("Interaction", "mean")]
            ),
            "interaction_std": float(
                network_stats.loc[network_id, ("Interaction", "std")]
            ),
        }

    evaluation_summary["network_statistics"] = network_stats_dict

    # Print evaluation summary
    logger.info("=== EVALUATION SUMMARY ===")
    logger.info(f"Network ID: {evaluation_summary['network_id']}")
    logger.info(f"Interactions: {evaluation_summary['num_interactions']}")
    logger.info(f"Plant species: {evaluation_summary['num_plant_species']}")
    logger.info(f"Pollinator species: {evaluation_summary['num_pollinator_species']}")
    logger.info(f"Total pairs: {evaluation_summary['num_pairs']}")
    logger.info(f"Model parameters: {evaluation_summary['model_parameters']:,}")
    logger.info(f"Network density: {evaluation_summary['network_density']:.4f}")
    logger.info(
        f"Avg interaction strength: {evaluation_summary['avg_interaction_strength']:.4f}"
    )
    logger.info(
        f"Interaction strength range: {evaluation_summary['min_interaction_strength']:.4f} - {evaluation_summary['max_interaction_strength']:.4f}"
    )
    logger.info(
        f"Test AUC: {evaluation_summary['test_auc']:.4f}, AP: {evaluation_summary['test_average_precision']:.4f}, Loss: {evaluation_summary['test_loss']:.4f}"
    )

    return evaluation_summary


def main():
    """Main test function."""
    logger.info("Starting pipeline test with network-focused sample")

    try:
        # Load data
        loader = EuPPollNetLoader("data/raw")

        # Use specified network ID
        network_id = "9_Heleno_Coimbra_2017_2017"
        logger.info(f"Using specified network ID: {network_id}")

        # Create network-focused sample
        data = create_network_sample(loader, network_id, max_interactions=300)

        # Test feature engineering
        plant_features, pollinator_features, pair_features, feature_engineer = (
            test_feature_engineering(data)
        )

        # Test GNN model
        model, trainer = test_gnn_model(
            plant_features, pollinator_features, pair_features, data
        )

        # Test model evaluation
        evaluation_summary = test_model_evaluation(
            model, trainer, plant_features, pollinator_features, pair_features, data
        )

        logger.info("=== Pipeline test completed successfully! ===")

        # Save evaluation summary to file
        import json

        output_file = f"results/test_evaluation_{network_id.replace('/', '_')}.json"
        Path("results").mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(evaluation_summary, f, indent=2, default=str)

        logger.info(f"Evaluation summary saved to: {output_file}")

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise


if __name__ == "__main__":
    main()
