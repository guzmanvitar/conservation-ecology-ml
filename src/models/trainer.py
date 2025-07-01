"""
Model Training for Plant-Pollinator Interaction Prediction

This module provides a comprehensive training pipeline for predicting
plant-pollinator interactions using machine learning models.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class InteractionModelTrainer:
    """
    Trainer for plant-pollinator interaction prediction models.

    Supports multiple model types with comprehensive evaluation
    and feature importance analysis.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model trainer.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.training_history = {}

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = "interaction_present",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training by splitting into train/test sets.

        Args:
            features_df: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info("Preparing data for training...")

        # Exclude non-feature columns
        exclude_cols = [
            "interaction_present",
            "interaction_strength",
            "plant_species",
            "pollinator_species",
            "network_id",
        ]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_names = feature_cols

        X = features_df[feature_cols].values
        y = features_df[target_col].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(
            f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples"
        )
        logger.info(f"Feature matrix shape: {X_train.shape[1]} features")
        logger.info(
            f"Target distribution - Train: {np.bincount(y_train.astype(int))}, Test: {np.bincount(y_test.astype(int))}"
        )

        return X_train, X_test, y_train, y_test, feature_cols

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = "random_forest",
        **kwargs,
    ) -> RandomForestClassifier:
        """
        Train a Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
            model_name: Name for the model
            **kwargs: Additional parameters for RandomForestClassifier

        Returns:
            Trained RandomForestClassifier
        """
        logger.info(f"Training Random Forest model: {model_name}")

        # Default parameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }

        # Update with provided parameters
        default_params.update(kwargs)

        # Train model
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)

        # Store model
        self.models[model_name] = model

        logger.info(f"Random Forest training completed")
        return model

    def evaluate_model(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model for logging

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Log results
        logger.info(f"Model {model_name} performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def cross_validate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        model_name: str = "model",
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to cross-validate
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            model_name: Name of the model

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation for {model_name}")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Cross-validate different metrics
        cv_results = {}
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        for metric in metrics:
            if metric == "roc_auc":
                scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            elif metric == "precision":
                scores = cross_val_score(model, X, y, cv=cv, scoring="precision")
            elif metric == "recall":
                scores = cross_val_score(model, X, y, cv=cv, scoring="recall")
            elif metric == "f1":
                scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
            else:  # accuracy
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

            cv_results[metric] = scores.tolist()

            logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_results

    def analyze_feature_importance(
        self,
        model,
        feature_names: List[str],
        model_name: str = "model",
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Analyze feature importance for a trained model.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to show

        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Analyzing feature importance for {model_name}")

        if hasattr(model, "feature_importances_"):
            importance_scores = model.feature_importances_
        else:
            logger.warning(
                f"Model {model_name} does not have feature_importances_ attribute"
            )
            return pd.DataFrame()

        # Create importance DataFrame
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_scores}
        ).sort_values("importance", ascending=False)

        # Show top features
        logger.info(f"Top {top_n} most important features:")
        for i, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def plot_feature_importance(
        self, importance_df: pd.DataFrame, model_name: str = "model", top_n: int = 20
    ):
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with feature importance scores
            model_name: Name of the model
            top_n: Number of top features to plot
        """
        if importance_df.empty:
            logger.warning("No feature importance data to plot")
            return

        plt.figure(figsize=(12, 8))

        top_features = importance_df.head(top_n)

        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        plot_path = self.models_dir / f"{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"Feature importance plot saved to {plot_path}")

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"
    ):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Interaction", "Interaction"],
            yticklabels=["No Interaction", "Interaction"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # Save plot
        plot_path = self.models_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"Confusion matrix plot saved to {plot_path}")

    def save_model(self, model, model_name: str = "model"):
        """
        Save a trained model to disk.

        Args:
            model: Trained model to save
            model_name: Name for the saved model
        """
        model_path = self.models_dir / f"{model_name}.joblib"

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "feature_names": self.feature_names,
            "model_type": type(model).__name__,
            "training_date": pd.Timestamp.now().isoformat(),
        }

        metadata_path = self.models_dir / f"{model_name}_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

    def load_model(self, model_name: str = "model"):
        """
        Load a trained model from disk.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model and metadata
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        metadata_path = self.models_dir / f"{model_name}_metadata.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        model = joblib.load(model_path)

        # Load metadata
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

        logger.info(f"Model loaded from {model_path}")

        return model, metadata

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        model_name: str = "random_forest",
        save_model: bool = True,
        **model_params,
    ) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.

        Args:
            features_df: DataFrame with features and target
            model_name: Name for the model
            save_model: Whether to save the trained model
            **model_params: Parameters for the model

        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting complete training pipeline for {model_name}")

        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(features_df)

        # Train model
        if model_name.startswith("random_forest"):
            model = self.train_random_forest(
                X_train, y_train, model_name, **model_params
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Evaluate on test set
        test_metrics = self.evaluate_model(model, X_test, y_test, model_name)

        # Cross-validation
        cv_results = self.cross_validate_model(
            model, X_train, y_train, model_name=model_name
        )

        # Feature importance
        importance_df = self.analyze_feature_importance(
            model, feature_names, model_name
        )

        # Plotting
        if not importance_df.empty:
            self.plot_feature_importance(importance_df, model_name)

        self.plot_confusion_matrix(y_test, model.predict(X_test), model_name)

        # Save model
        if save_model:
            self.save_model(model, model_name)

        # Compile results
        results = {
            "model_name": model_name,
            "test_metrics": test_metrics,
            "cv_results": cv_results,
            "feature_importance": importance_df,
            "feature_names": feature_names,
            "model": model,
        }

        logger.info(f"Training pipeline completed for {model_name}")
        return results


def create_sample_training_data() -> pd.DataFrame:
    """
    Create sample data for testing the training pipeline.

    Returns:
        DataFrame with sample features and target
    """
    logger.info("Creating sample training data")

    # Import feature engineering
    import sys

    sys.path.append("src")
    from features.engineering import create_sample_features

    # Create sample features
    features = create_sample_features()

    # Ensure we have some positive interactions
    features["interaction_present"] = [1, 1, 0]  # Force some positive examples

    return features


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create trainer
    trainer = InteractionModelTrainer()

    # Load sample data
    features = create_sample_training_data()

    # Train and evaluate
    results = trainer.train_and_evaluate(features, model_name="sample_random_forest")

    print("Training completed!")
    print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 score: {results['test_metrics']['f1']:.4f}")
