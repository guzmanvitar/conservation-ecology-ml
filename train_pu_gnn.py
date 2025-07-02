#!/usr/bin/env python3
"""
Main training pipeline for PU Learning + GNN for plant-pollinator interaction prediction.

This script orchestrates the entire pipeline:
1. Load and preprocess data
2. Create features for PU learning
3. Build bipartite graph
4. Train PU GNN model
5. Evaluate and save results
"""

import argparse
import json
import logging
import sys

# Removed datetime import - not using it
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.loader import EuPPollNetLoader
from features.pu_features import PUFeatureEngineer
from models.pu_gnn import BipartiteGNN, PUGNNTrainer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PU Learning + GNN for plant-pollinator interactions"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pu_gnn",
        help="Directory to save results",
    )

    # Model arguments
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of GNN layers"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--conv_type",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "sage"],
        help="Type of graph convolution",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Validation split ratio"
    )

    # PU Learning arguments
    parser.add_argument(
        "--prior", type=float, default=0.1, help="Prior probability of positive class"
    )
    parser.add_argument(
        "--beta", type=float, default=0.0, help="Beta parameter for PU loss"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Gamma parameter for PU loss"
    )
    parser.add_argument(
        "--num_unlabeled_samples",
        type=int,
        default=10000,
        help="Number of unlabeled pairs to sample",
    )

    # Feature arguments
    parser.add_argument(
        "--use_traits",
        action="store_true",
        default=True,
        help="Use species trait features",
    )
    parser.add_argument(
        "--use_network_features",
        action="store_true",
        default=True,
        help="Use network topology features",
    )
    parser.add_argument(
        "--use_temporal_features",
        action="store_true",
        default=True,
        help="Use temporal features",
    )
    parser.add_argument(
        "--use_spatial_features",
        action="store_true",
        default=True,
        help="Use spatial features",
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--save_model", action="store_true", default=True, help="Save trained model"
    )
    parser.add_argument(
        "--network_id",
        type=str,
        default="9_Heleno_Coimbra_2017_2017",
        help="Network_id to use (default: 9_Heleno_Coimbra_2017_2017, use 'all' for all networks)",
    )
    # Removed wandb argument - not using it

    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup device for training."""
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    return device


def load_data(data_dir: str, args=None) -> Dict[str, pd.DataFrame]:
    """Load all data files."""
    logger.info("Loading data...")

    loader = EuPPollNetLoader(data_dir)
    data = loader.load_all_data()

    # Filter to a single network if requested
    if args is not None and hasattr(args, "network_id") and args.network_id != "all":
        network_id = args.network_id
        for key, df in data.items():
            if df is not None and "Network_id" in df.columns:
                data[key] = df[df["Network_id"] == network_id].copy()
        logger.info(f"Filtered data to network_id={network_id}")
    else:
        logger.info("Using all networks.")

    logger.info("Data loaded successfully:")
    for key, df in data.items():
        if df is not None:
            logger.info(f"  {key}: {df.shape}")

    return data


def create_features(
    data: Dict[str, pd.DataFrame], args: argparse.Namespace
) -> Dict[str, pd.DataFrame]:
    """Create features for PU learning."""
    logger.info("Creating features...")

    feature_engineer = PUFeatureEngineer(
        use_traits=args.use_traits,
        use_network_features=args.use_network_features,
        use_temporal_features=args.use_temporal_features,
        use_spatial_features=args.use_spatial_features,
    )

    # Create features
    plant_features, pollinator_features, pair_features = feature_engineer.fit_transform(
        plant_taxonomy=data["plants_taxonomy"],
        pollinator_taxonomy=data["pollinators_taxonomy"],
        interactions=data["interactions"],
        flower_counts=data.get("flower_counts"),
    )

    features = {
        "plant_features": plant_features,
        "pollinator_features": pollinator_features,
        "pair_features": pair_features,
        "feature_engineer": feature_engineer,
    }

    logger.info("Features created successfully:")
    logger.info(f"  Plant features: {plant_features.shape}")
    logger.info(f"  Pollinator features: {pollinator_features.shape}")
    logger.info(f"  Pair features: {pair_features.shape}")

    return features


def prepare_training_data(
    features: Dict[str, pd.DataFrame],
    data: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Prepare data for training."""
    logger.info("Preparing training data...")

    # Get positive pairs (observed interactions)
    interactions = data["interactions"]
    positive_pairs = list(
        zip(
            interactions["Plant_accepted_name"],
            interactions["Pollinator_accepted_name"],
        )
    )
    positive_pairs = list(set(positive_pairs))  # Remove duplicates

    logger.info(f"Found {len(positive_pairs)} unique positive pairs")

    # Create plant and pollinator mappings
    plant_to_idx = {
        plant: idx for idx, plant in enumerate(features["plant_features"].index)
    }
    pollinator_to_idx = {
        pollinator: idx
        for idx, pollinator in enumerate(features["pollinator_features"].index)
    }

    # Convert positive pairs to indices
    positive_pairs_idx = []
    for plant, pollinator in positive_pairs:
        if plant in plant_to_idx and pollinator in pollinator_to_idx:
            plant_idx = plant_to_idx[plant]
            pollinator_idx = pollinator_to_idx[pollinator]
            positive_pairs_idx.append((plant_idx, pollinator_idx))

    logger.info(f"Converted {len(positive_pairs_idx)} positive pairs to indices")

    # Sample unlabeled pairs
    trainer = PUGNNTrainer(
        model=None,  # Will be set later
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    unlabeled_pairs = trainer.sample_unlabeled_pairs(
        interactions=interactions,
        plant_to_idx=plant_to_idx,
        pollinator_to_idx=pollinator_to_idx,
        num_samples=args.num_unlabeled_samples,
    )

    logger.info(f"Sampled {len(unlabeled_pairs)} unlabeled pairs")

    # Create bipartite graph
    graph_data, _, _ = trainer.create_bipartite_graph(
        interactions=interactions,
        plant_features=features["plant_features"],
        pollinator_features=features["pollinator_features"],
    )

    logger.info(
        f"Created bipartite graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges"
    )

    training_data = {
        "graph_data": graph_data,
        "positive_pairs": positive_pairs_idx,
        "unlabeled_pairs": unlabeled_pairs,
        "plant_to_idx": plant_to_idx,
        "pollinator_to_idx": pollinator_to_idx,
        "plant_features": features["plant_features"],
        "pollinator_features": features["pollinator_features"],
    }

    return training_data


def create_model(
    features: Dict[str, pd.DataFrame], args: argparse.Namespace
) -> BipartiteGNN:
    """Create the GNN model."""
    logger.info("Creating model...")

    plant_features_dim = features["plant_features"].shape[1]
    pollinator_features_dim = features["pollinator_features"].shape[1]

    model = BipartiteGNN(
        plant_features=plant_features_dim,
        pollinator_features=pollinator_features_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)"
    )
    logger.info(
        f"Model architecture: {args.conv_type.upper()} with {args.num_layers} layers, "
        f"hidden_dim={args.hidden_dim}"
    )

    return model


def train_model(
    model: BipartiteGNN, training_data: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Train the model."""
    logger.info("Starting training...")

    # Create trainer
    trainer = PUGNNTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Train model
    losses = trainer.train(
        data=training_data["graph_data"],
        positive_pairs=training_data["positive_pairs"],
        unlabeled_pairs=training_data["unlabeled_pairs"],
        num_epochs=args.num_epochs,
        val_split=args.val_split,
    )

    # Final evaluation
    val_metrics = trainer.evaluate(
        data=training_data["graph_data"],
        test_positive_pairs=training_data["positive_pairs"][
            : int(len(training_data["positive_pairs"]) * args.val_split)
        ],
        test_unlabeled_pairs=training_data["unlabeled_pairs"][
            : int(len(training_data["unlabeled_pairs"]) * args.val_split)
        ],
    )

    results = {
        "trainer": trainer,
        "losses": losses,
        "final_metrics": val_metrics,
        "model": model,
    }

    logger.info("Training completed!")
    logger.info(f"Final metrics: {val_metrics}")

    return results


def save_results(
    results: Dict[str, Any],
    features: Dict[str, pd.DataFrame],
    training_data: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: str,
):
    """Save results and model."""
    logger.info("Saving results...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    if args.save_model:
        model_path = output_path / "model.pt"
        torch.save(
            {
                "model_state_dict": results["model"].state_dict(),
                "model_config": {
                    "plant_features": features["plant_features"].shape[1],
                    "pollinator_features": features["pollinator_features"].shape[1],
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "conv_type": args.conv_type,
                },
                "feature_engineer": features["feature_engineer"],
                "plant_to_idx": training_data["plant_to_idx"],
                "pollinator_to_idx": training_data["pollinator_to_idx"],
            },
            model_path,
        )
        logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_path / "training_history.json"
    history = {
        "losses": results["losses"],
        "final_metrics": results["final_metrics"],
        "training_config": vars(args),
        "data_info": {
            "num_plants": len(features["plant_features"]),
            "num_pollinators": len(features["pollinator_features"]),
            "num_positive_pairs": len(training_data["positive_pairs"]),
            "num_unlabeled_pairs": len(training_data["unlabeled_pairs"]),
            "graph_nodes": training_data["graph_data"].num_nodes,
            "graph_edges": training_data["graph_data"].num_edges,
        },
    }

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info(f"Training history saved to {history_path}")

    # Save feature importance (if applicable)
    # TODO: Implement feature importance analysis
    logger.info("Feature importance analysis not implemented yet")

    logger.info(f"Results saved to {output_path}")


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup device
    args.device = setup_device(args.device)

    # Not using wandb for this run

    try:
        # Load data
        data = load_data(args.data_dir, args)

        # Create features
        features = create_features(data, args)

        # Prepare training data
        training_data = prepare_training_data(features, data, args)

        # Create model
        model = create_model(features, args)

        # Train model
        results = train_model(model, training_data, args)

        # Save results
        save_results(results, features, training_data, args, args.output_dir)

        logger.info("Pipeline completed successfully!")

        # Not using wandb for this run

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
