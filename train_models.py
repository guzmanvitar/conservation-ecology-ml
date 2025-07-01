#!/usr/bin/env python3
"""
Main training script for plant-pollinator interaction prediction.

This script loads the EuPPollNet data, creates features, and trains
machine learning models to predict plant-pollinator interactions.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append("src")

from data.loader import EuPPollNetLoader
from features.engineering import InteractionFeatureEngineer
from models.trainer import InteractionModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""

    print("=" * 80)
    print("PLANT-POLLINATOR INTERACTION PREDICTION TRAINING")
    print("=" * 80)

    # Step 1: Load data
    print("\n1. Loading EuPPollNet data...")
    try:
        loader = EuPPollNetLoader("data/raw")
        data = loader.load_all_data()
        interactions = data["interactions"]

        print(f"✅ Loaded {len(interactions):,} interactions")
        print(f"✅ Networks: {interactions['Network_id_full'].nunique():,}")
        print(f"✅ Plant species: {interactions['Plant_accepted_name'].nunique():,}")
        print(
            f"✅ Pollinator species: {interactions['Pollinator_accepted_name'].nunique():,}"
        )

    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return

        # Step 2: Create features
    print("\n2. Creating features...")
    try:
        # Use a sample for training (5,000 interactions to keep it manageable)
        sample_size = min(5000, len(interactions))
        sample_interactions = interactions.sample(n=sample_size, random_state=42)

        print(f"Using {len(sample_interactions):,} positive interactions as base")

        # Create feature engineer
        engineer = InteractionFeatureEngineer("data/raw")

        # Generate negative examples to create balanced dataset
        balanced_interactions = engineer.generate_negative_examples(
            sample_interactions, negative_ratio=1.0
        )

        print(
            f"Created balanced dataset with {len(balanced_interactions):,} total examples"
        )

        # Create features
        features = engineer.create_all_features(balanced_interactions)

        print(f"✅ Created {len(features.columns)} features")
        print(f"✅ Features shape: {features.shape}")

        # Check target distribution
        target_dist = features["interaction_present"].value_counts()
        print(f"✅ Target distribution:")
        print(
            f"   Interactions present: {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(features)*100:.1f}%)"
        )
        print(
            f"   No interactions: {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(features)*100:.1f}%)"
        )

    except Exception as e:
        logger.error(f"❌ Error creating features: {e}")
        return

    # Step 3: Train models
    print("\n3. Training models...")
    try:
        # Create trainer
        trainer = InteractionModelTrainer("models")

        # Train Random Forest with different configurations
        models_to_train = [
            {
                "name": "random_forest_basic",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "random_forest_deep",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 20,
                    "min_samples_split": 2,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "random_forest_wide",
                "params": {
                    "n_estimators": 300,
                    "max_depth": 15,
                    "min_samples_leaf": 1,
                    "class_weight": "balanced",
                },
            },
        ]

        results = {}

        for model_config in models_to_train:
            print(f"\n--- Training {model_config['name']} ---")

            try:
                result = trainer.train_and_evaluate(
                    features,
                    model_name=model_config["name"],
                    save_model=True,
                    **model_config["params"],
                )

                results[model_config["name"]] = result

                print(f"✅ {model_config['name']} training completed!")
                print(f"   Test accuracy: {result['test_metrics']['accuracy']:.4f}")
                print(f"   Test F1 score: {result['test_metrics']['f1']:.4f}")
                print(f"   Test ROC AUC: {result['test_metrics']['roc_auc']:.4f}")

            except Exception as e:
                logger.error(f"❌ Error training {model_config['name']}: {e}")
                continue

    except Exception as e:
        logger.error(f"❌ Error in training pipeline: {e}")
        return

    # Step 4: Compare models
    print("\n4. Model comparison...")
    if results:
        print("\nModel Performance Summary:")
        print("-" * 80)
        print(
            f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'ROC AUC':<10} {'Precision':<10} {'Recall':<10}"
        )
        print("-" * 80)

        for model_name, result in results.items():
            metrics = result["test_metrics"]
            print(
                f"{model_name:<25} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
                f"{metrics['roc_auc']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f}"
            )

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]["test_metrics"]["f1"])
        print(
            f"\n🏆 Best model by F1 score: {best_model[0]} (F1: {best_model[1]['test_metrics']['f1']:.4f})"
        )

        # Show feature importance for best model
        if not best_model[1]["feature_importance"].empty:
            print(f"\nTop 10 most important features for {best_model[0]}:")
            top_features = best_model[1]["feature_importance"].head(10)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")

    # Step 5: Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"✅ Data loaded: {len(interactions):,} interactions")
    print(f"✅ Features created: {len(features.columns)} features")
    print(f"✅ Models trained: {len(results)} models")
    print(f"✅ Models saved to: models/")

    if results:
        best_f1 = max(r["test_metrics"]["f1"] for r in results.values())
        best_auc = max(r["test_metrics"]["roc_auc"] for r in results.values())
        print(f"✅ Best F1 score: {best_f1:.4f}")
        print(f"✅ Best ROC AUC: {best_auc:.4f}")

    print("\n🎉 Training pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Check the 'models/' directory for saved models")
    print("2. Review feature importance plots")
    print("3. Use the best model for predictions")
    print("4. Consider hyperparameter tuning for further improvement")


if __name__ == "__main__":
    main()
