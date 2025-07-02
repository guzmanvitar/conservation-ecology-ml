"""
PU Learning + GNN for Plant-Pollinator Interaction Prediction

This module implements Positive-Unlabeled Learning with Graph Neural Networks
for predicting missing plant-pollinator interactions in ecological networks.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

logger = logging.getLogger(__name__)


class PULoss(nn.Module):
    """
    Positive-Unlabeled Loss implementation.

    Based on the non-negative PU loss (nnPU) from Kiryo et al. 2017.
    """

    def __init__(self, prior: float = 0.1, beta: float = 0.0, gamma: float = 1.0):
        """
        Initialize PU Loss.

        Args:
            prior: Prior probability of positive class
            beta: Weight for negative risk
            gamma: Weight for positive risk
        """
        super().__init__()
        self.prior = prior
        self.beta = beta
        self.gamma = gamma

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PU loss.

        Args:
            outputs: Model predictions (logits)
            targets: Binary labels (1 for positive, 0 for unlabeled)
            positive_mask: Boolean mask indicating positive samples

        Returns:
            PU loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(outputs)

        # Positive samples
        positive_probs = probs[positive_mask]
        positive_risk = -torch.log(positive_probs + 1e-8).mean()

        # Unlabeled samples
        unlabeled_mask = ~positive_mask
        unlabeled_probs = probs[unlabeled_mask]

        # Risk for unlabeled samples
        unlabeled_risk = torch.log(1 - unlabeled_probs + 1e-8).mean()

        # Non-negative PU loss
        negative_risk = unlabeled_risk - self.prior * positive_risk

        if negative_risk < -self.beta:
            negative_risk = -self.beta

        total_loss = self.gamma * positive_risk + negative_risk

        return total_loss


class BipartiteGNN(nn.Module):
    """
    Bipartite Graph Neural Network for plant-pollinator interaction prediction.

    This model handles the bipartite nature of plant-pollinator networks
    with separate encoders for plants and pollinators.
    """

    def __init__(
        self,
        plant_features: int,
        pollinator_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_type: str = "gcn",
    ):
        """
        Initialize Bipartite GNN.

        Args:
            plant_features: Number of plant node features
            pollinator_features: Number of pollinator node features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of graph convolution ("gcn", "gat", "sage")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type

        # Plant encoder
        self.plant_encoder = nn.Sequential(
            nn.Linear(plant_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pollinator encoder
        self.pollinator_encoder = nn.Sequential(
            nn.Linear(pollinator_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif conv_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            self.convs.append(conv)

        # Interaction predictor
        self.interaction_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_nodes(
        self, plant_features: torch.Tensor, pollinator_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode plant and pollinator features.

        Args:
            plant_features: Plant node features
            pollinator_features: Pollinator node features

        Returns:
            Combined node features
        """
        plant_encoded = self.plant_encoder(plant_features)
        pollinator_encoded = self.pollinator_encoder(pollinator_features)

        # Combine features
        combined_features = torch.cat([plant_encoded, pollinator_encoded], dim=0)
        return combined_features

    def forward(self, data: Data, num_plants: int = None) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            data: PyTorch Geometric Data object
            num_plants: Number of plant nodes (optional, will be inferred if not provided)

        Returns:
            Node embeddings
        """
        x = data.x
        edge_index = data.edge_index

        # Split features into plant and pollinator parts
        if num_plants is None:
            num_plants = x.shape[0] // 2  # Approximate split
        plant_features = x[:num_plants]
        pollinator_features = x[num_plants:]

        # Encode features using separate encoders
        plant_encoded = self.plant_encoder(plant_features)
        pollinator_encoded = self.pollinator_encoder(pollinator_features)

        # Combine encoded features
        x = torch.cat([plant_encoded, pollinator_encoded], dim=0)

        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def predict_interaction(
        self, plant_emb: torch.Tensor, pollinator_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict interaction probability between plant and pollinator.

        Args:
            plant_emb: Plant embedding
            pollinator_emb: Pollinator embedding

        Returns:
            Interaction probability logits
        """
        # Concatenate embeddings
        combined = torch.cat([plant_emb, pollinator_emb], dim=-1)

        # Predict interaction
        logits = self.interaction_predictor(combined)
        return logits


class PUGNNTrainer:
    """
    Trainer for PU Learning with GNNs.
    """

    def __init__(
        self,
        model: BipartiteGNN,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize PU GNN trainer.

        Args:
            model: Bipartite GNN model
            device: Device to use for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.pu_loss = PULoss()

    def create_bipartite_graph(
        self,
        interactions: pd.DataFrame,
        plant_features: pd.DataFrame,
        pollinator_features: pd.DataFrame,
    ) -> Data:
        """
        Create bipartite graph from interaction data.

        Args:
            interactions: DataFrame with plant-pollinator interactions
            plant_features: DataFrame with plant features
            pollinator_features: DataFrame with pollinator features

        Returns:
            PyTorch Geometric Data object
        """
        # Create node mappings
        plant_to_idx = {plant: idx for idx, plant in enumerate(plant_features.index)}
        pollinator_to_idx = {
            pollinator: idx for idx, pollinator in enumerate(pollinator_features.index)
        }

        # Build edges
        edges = []
        for _, row in interactions.iterrows():
            plant_idx = plant_to_idx.get(row["Plant_accepted_name"])
            pollinator_idx = pollinator_to_idx.get(row["Pollinator_accepted_name"])
            if plant_idx is not None and pollinator_idx is not None:
                edges.append([plant_idx, len(plant_to_idx) + pollinator_idx])
                # Add edge from pollinator to plant (undirected)
                edges.append([len(plant_to_idx) + pollinator_idx, plant_idx])

        logger.debug(f"Number of edges before creating edge_index: {len(edges)}")
        if len(edges) == 0:
            logger.warning(
                "No edges were created in the bipartite graph! Check your mappings and data."
            )
            raise ValueError(
                "No edges were created in the bipartite graph. Cannot proceed."
            )

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Create node features (only numeric columns)
        plant_numeric = plant_features.select_dtypes(include=[np.number])
        pollinator_numeric = pollinator_features.select_dtypes(include=[np.number])

        logger.debug(f"plant_numeric shape: {plant_numeric.shape}")
        logger.debug(f"pollinator_numeric shape: {pollinator_numeric.shape}")
        logger.debug(f"plant_features index sample: {list(plant_features.index)[:5]}")
        logger.debug(
            f"pollinator_features index sample: {list(pollinator_features.index)[:5]}"
        )
        logger.debug(f"plant_to_idx sample: {list(plant_to_idx.items())[:5]}")
        logger.debug(f"pollinator_to_idx sample: {list(pollinator_to_idx.items())[:5]}")

        plant_feat = torch.tensor(plant_numeric.values, dtype=torch.float)
        pollinator_feat = torch.tensor(pollinator_numeric.values, dtype=torch.float)

        # Combine features
        x = torch.cat([plant_feat, pollinator_feat], dim=0)

        logger.debug(f"Combined node feature shape: {x.shape}")
        logger.debug(f"Number of edges: {edge_index.shape[1]}")
        if edge_index.shape[1] > 0:
            logger.debug(f"Sample edge_index: {edge_index[:, :5]}")

        # Create graph
        data = Data(x=x, edge_index=edge_index)

        return data, plant_to_idx, pollinator_to_idx, len(plant_to_idx)

    def sample_unlabeled_pairs(
        self,
        interactions: pd.DataFrame,
        plant_to_idx: Dict,
        pollinator_to_idx: Dict,
        num_samples: int = 10000,
    ) -> List[Tuple[int, int]]:
        """
        Sample unlabeled plant-pollinator pairs.

        Args:
            interactions: Observed interactions
            plant_to_idx: Plant to index mapping
            pollinator_to_idx: Pollinator to index mapping
            num_samples: Number of unlabeled pairs to sample

        Returns:
            List of unlabeled pairs
        """
        # Get all possible pairs
        all_plants = list(plant_to_idx.keys())
        all_pollinators = list(pollinator_to_idx.keys())

        # Create set of observed pairs
        observed_pairs = set()
        for _, row in interactions.iterrows():
            plant = row["Plant_accepted_name"]
            pollinator = row["Pollinator_accepted_name"]
            if plant in plant_to_idx and pollinator in pollinator_to_idx:
                observed_pairs.add((plant, pollinator))

        # Sample unlabeled pairs
        unlabeled_pairs = []
        attempts = 0
        max_attempts = num_samples * 10

        while len(unlabeled_pairs) < num_samples and attempts < max_attempts:
            plant = np.random.choice(all_plants)
            pollinator = np.random.choice(all_pollinators)

            if (plant, pollinator) not in observed_pairs:
                plant_idx = plant_to_idx[plant]
                pollinator_idx = pollinator_to_idx[pollinator]
                unlabeled_pairs.append((plant_idx, pollinator_idx))

            attempts += 1

        return unlabeled_pairs

    def train_epoch(
        self,
        data: Data,
        positive_pairs: List[Tuple[int, int]],
        unlabeled_pairs: List[Tuple[int, int]],
        num_plants: int = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            data: Graph data
            positive_pairs: List of positive plant-pollinator pairs
            unlabeled_pairs: List of unlabeled plant-pollinator pairs

        Returns:
            Average loss for the epoch
        """
        self.model.train()

        # Move data to device
        data = data.to(self.device)

        # Get node embeddings
        node_embeddings = self.model(data, num_plants=num_plants)

        # Prepare training data
        all_pairs = positive_pairs + unlabeled_pairs
        labels = [1] * len(positive_pairs) + [0] * len(unlabeled_pairs)

        # Convert to tensors
        pair_tensor = torch.tensor(all_pairs, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        # Get embeddings for pairs
        plant_embs = node_embeddings[pair_tensor[:, 0]]
        pollinator_embs = node_embeddings[pair_tensor[:, 1]]

        # Predict interactions
        logits = self.model.predict_interaction(plant_embs, pollinator_embs).squeeze()

        # Create positive mask
        positive_mask = labels_tensor == 1

        # Compute loss
        loss = self.pu_loss(logits, labels_tensor, positive_mask)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        data: Data,
        test_positive_pairs: List[Tuple[int, int]],
        test_unlabeled_pairs: List[Tuple[int, int]],
        num_plants: int = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            data: Graph data
            test_positive_pairs: Test positive pairs
            test_unlabeled_pairs: Test unlabeled pairs

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            data = data.to(self.device)
            node_embeddings = self.model(data, num_plants=num_plants)

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
            probs = torch.sigmoid(logits)

            # Compute metrics
            auc = roc_auc_score(labels, probs.cpu().numpy())
            ap = average_precision_score(labels, probs.cpu().numpy())

            return {
                "auc": auc,
                "average_precision": ap,
                "loss": self.pu_loss(logits, labels_tensor, labels_tensor == 1).item(),
            }

    def train(
        self,
        data: Data,
        positive_pairs: List[Tuple[int, int]],
        unlabeled_pairs: List[Tuple[int, int]],
        num_epochs: int = 100,
        val_split: float = 0.2,
        num_plants: int = None,
    ) -> List[float]:
        """
        Train the model.

        Args:
            data: Graph data
            positive_pairs: Positive plant-pollinator pairs
            unlabeled_pairs: Unlabeled plant-pollinator pairs
            num_epochs: Number of training epochs
            val_split: Validation split ratio

        Returns:
            List of training losses
        """
        # Split data
        num_val = max(
            1, int(len(positive_pairs) * val_split)
        )  # Ensure at least 1 validation sample
        train_pos = positive_pairs[num_val:]
        val_pos = positive_pairs[:num_val]

        num_val_unlabeled = max(
            1, int(len(unlabeled_pairs) * val_split)
        )  # Ensure at least 1 validation sample
        train_unlabeled = unlabeled_pairs[num_val_unlabeled:]
        val_unlabeled = unlabeled_pairs[:num_val_unlabeled]

        losses = []

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(
                data, train_pos, train_unlabeled, num_plants=num_plants
            )

            # Validation
            if epoch % 10 == 0:
                val_metrics = self.evaluate(
                    data, val_pos, val_unlabeled, num_plants=num_plants
                )
                logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )

            losses.append(train_loss)

        return losses
