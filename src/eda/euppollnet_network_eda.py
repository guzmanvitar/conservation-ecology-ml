# %%
"""
EuPPollNet Network Analysis EDA
===============================

This script performs comprehensive exploratory data analysis on the EuPPollNet dataset,
with special focus on network metrics and visualization techniques for large networks.

The EuPPollNet dataset contains 623,476 interactions across 1,864 networks from 23 countries.
"""

import warnings

warnings.filterwarnings("ignore")

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
from plotly.subplots import make_subplots
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Import our data loader
import sys

sys.path.append("src")
from data.loader import EuPPollNetLoader

# %%
# Load the EuPPollNet dataset
print("Loading EuPPollNet dataset...")
loader = EuPPollNetLoader("data/raw")
data = loader.load_all_data()

interactions = data["interactions"]
plants_taxonomy = data["plants_taxonomy"]
pollinators_taxonomy = data["pollinators_taxonomy"]
metadata = data["metadata"]
flower_counts = data["flower_counts"]

print(f"Dataset loaded successfully!")
print(f"Interactions: {len(interactions):,}")
print(f"Networks: {interactions['Network_id_full'].nunique():,}")
print(f"Countries: {interactions['Country'].nunique()}")
print(f"Plant species: {interactions['Plant_accepted_name'].nunique():,}")
print(f"Pollinator species: {interactions['Pollinator_accepted_name'].nunique():,}")

# %%
# Basic dataset overview
print("=" * 80)
print("EUPOLLNET DATASET OVERVIEW")
print("=" * 80)

print(f"\n📊 INTERACTIONS:")
print(f"   Total interactions: {len(interactions):,}")
print(f"   Date range: {interactions['Date'].min()} to {interactions['Date'].max()}")
print(f"   Years covered: {interactions['Year'].min()}-{interactions['Year'].max()}")
print(f"   Unique networks: {interactions['Network_id_full'].nunique():,}")
print(f"   Unique studies: {interactions['Study_id'].nunique():,}")

print(f"\n🌍 GEOGRAPHICAL COVERAGE:")
print(f"   Countries: {interactions['Country'].nunique()}")
print(f"   Countries: {sorted(interactions['Country'].unique())}")

print(f"\n🌱 PLANT DIVERSITY:")
print(f"   Plant species: {interactions['Plant_accepted_name'].nunique():,}")
print(f"   Plant genera: {interactions['Plant_genus'].nunique():,}")
print(f"   Plant families: {interactions['Plant_family'].nunique():,}")

print(f"\n🐝 POLLINATOR DIVERSITY:")
print(f"   Pollinator species: {interactions['Pollinator_accepted_name'].nunique():,}")
print(f"   Pollinator genera: {interactions['Pollinator_genus'].nunique():,}")
print(f"   Pollinator families: {interactions['Pollinator_family'].nunique():,}")
print(f"   Pollinator orders: {interactions['Pollinator_order'].nunique():,}")

print(f"\n📈 INTERACTION STRENGTH:")
print(f"   Mean interaction strength: {interactions['Interaction'].mean():.4f}")
print(f"   Max interaction strength: {interactions['Interaction'].max():.4f}")
print(f"   Min interaction strength: {interactions['Interaction'].min():.4f}")
print(f"   Total interaction strength: {interactions['Interaction'].sum():,.2f}")

# %%
# Network-level statistics
print("\n" + "=" * 80)
print("NETWORK-LEVEL STATISTICS")
print("=" * 80)

# Calculate network-level metrics
network_stats = (
    interactions.groupby("Network_id_full")
    .agg(
        {
            "Plant_accepted_name": "nunique",
            "Pollinator_accepted_name": "nunique",
            "Interaction": ["sum", "mean", "count"],
            "Country": "first",
            "Year": "first",
            "Study_id": "first",
        }
    )
    .round(4)
)

network_stats.columns = [
    "plant_species",
    "pollinator_species",
    "total_strength",
    "mean_strength",
    "interaction_count",
    "country",
    "year",
    "study_id",
]
network_stats = network_stats.reset_index()

print(f"Network statistics calculated for {len(network_stats)} networks")

print(f"\n📊 NETWORK SIZE DISTRIBUTION:")
print(f"   Mean plant species per network: {network_stats['plant_species'].mean():.1f}")
print(
    f"   Mean pollinator species per network: {network_stats['pollinator_species'].mean():.1f}"
)
print(
    f"   Mean interactions per network: {network_stats['interaction_count'].mean():.1f}"
)
print(f"   Largest network: {network_stats['interaction_count'].max():,} interactions")
print(f"   Smallest network: {network_stats['interaction_count'].min():,} interactions")

# %%
# Visualize network size distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Network Size Distributions", fontsize=16, fontweight="bold")

# Plant species per network
axes[0, 0].hist(
    network_stats["plant_species"],
    bins=30,
    alpha=0.7,
    color="lightgreen",
    edgecolor="black",
)
axes[0, 0].set_title("Plant Species per Network")
axes[0, 0].set_xlabel("Number of Plant Species")
axes[0, 0].set_ylabel("Number of Networks")
axes[0, 0].axvline(
    network_stats["plant_species"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {network_stats["plant_species"].mean():.1f}',
)
axes[0, 0].legend()

# Pollinator species per network
axes[0, 1].hist(
    network_stats["pollinator_species"],
    bins=30,
    alpha=0.7,
    color="orange",
    edgecolor="black",
)
axes[0, 1].set_title("Pollinator Species per Network")
axes[0, 1].set_xlabel("Number of Pollinator Species")
axes[0, 1].set_ylabel("Number of Networks")
axes[0, 1].axvline(
    network_stats["pollinator_species"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {network_stats["pollinator_species"].mean():.1f}',
)
axes[0, 1].legend()

# Interactions per network
axes[1, 0].hist(
    network_stats["interaction_count"],
    bins=30,
    alpha=0.7,
    color="skyblue",
    edgecolor="black",
)
axes[1, 0].set_title("Interactions per Network")
axes[1, 0].set_xlabel("Number of Interactions")
axes[1, 0].set_ylabel("Number of Networks")
axes[1, 0].axvline(
    network_stats["interaction_count"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {network_stats["interaction_count"].mean():.1f}',
)
axes[1, 0].legend()

# Network size vs interaction count
axes[1, 1].scatter(
    network_stats["plant_species"] * network_stats["pollinator_species"],
    network_stats["interaction_count"],
    alpha=0.6,
    color="purple",
)
axes[1, 1].set_title("Network Size vs Interaction Count")
axes[1, 1].set_xlabel("Plant Species × Pollinator Species")
axes[1, 1].set_ylabel("Number of Interactions")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Geographic distribution analysis
print("\n" + "=" * 80)
print("GEOGRAPHIC DISTRIBUTION ANALYSIS")
print("=" * 80)

country_stats = (
    interactions.groupby("Country")
    .agg(
        {
            "Network_id_full": "nunique",
            "Plant_accepted_name": "nunique",
            "Pollinator_accepted_name": "nunique",
            "Interaction": "sum",
        }
    )
    .round(2)
)

country_stats.columns = [
    "networks",
    "plant_species",
    "pollinator_species",
    "total_interactions",
]
country_stats = country_stats.sort_values("networks", ascending=False)

print("Top 10 countries by number of networks:")
print(country_stats.head(10))

# %%
# Geographic visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Geographic Distribution of EuPPollNet Data", fontsize=16, fontweight="bold"
)

# Networks per country
top_countries = country_stats.head(15)
axes[0, 0].barh(range(len(top_countries)), top_countries["networks"], color="lightblue")
axes[0, 0].set_yticks(range(len(top_countries)))
axes[0, 0].set_yticklabels(top_countries.index)
axes[0, 0].set_title("Number of Networks per Country (Top 15)")
axes[0, 0].set_xlabel("Number of Networks")

# Plant species per country
axes[0, 1].barh(
    range(len(top_countries)), top_countries["plant_species"], color="lightgreen"
)
axes[0, 1].set_yticks(range(len(top_countries)))
axes[0, 1].set_yticklabels(top_countries.index)
axes[0, 1].set_title("Plant Species per Country (Top 15)")
axes[0, 1].set_xlabel("Number of Plant Species")

# Pollinator species per country
axes[1, 0].barh(
    range(len(top_countries)), top_countries["pollinator_species"], color="orange"
)
axes[1, 0].set_yticks(range(len(top_countries)))
axes[1, 0].set_yticklabels(top_countries.index)
axes[1, 0].set_title("Pollinator Species per Country (Top 15)")
axes[1, 0].set_xlabel("Number of Pollinator Species")

# Total interactions per country
axes[1, 1].barh(
    range(len(top_countries)), top_countries["total_interactions"], color="purple"
)
axes[1, 1].set_yticks(range(len(top_countries)))
axes[1, 1].set_yticklabels(top_countries.index)
axes[1, 1].set_title("Total Interactions per Country (Top 15)")
axes[1, 1].set_xlabel("Total Interaction Strength")

plt.tight_layout()
plt.show()

# %%
# Temporal analysis
print("\n" + "=" * 80)
print("TEMPORAL ANALYSIS")
print("=" * 80)

# Yearly trends
yearly_stats = (
    interactions.groupby("Year")
    .agg(
        {
            "Network_id_full": "nunique",
            "Plant_accepted_name": "nunique",
            "Pollinator_accepted_name": "nunique",
            "Interaction": "sum",
        }
    )
    .round(2)
)

yearly_stats.columns = [
    "networks",
    "plant_species",
    "pollinator_species",
    "total_interactions",
]

print("Yearly trends:")
print(yearly_stats)

# %%
# Temporal visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Temporal Trends in EuPPollNet Data", fontsize=16, fontweight="bold")

# Networks per year
axes[0, 0].plot(
    yearly_stats.index, yearly_stats["networks"], marker="o", linewidth=2, markersize=6
)
axes[0, 0].set_title("Networks per Year")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Number of Networks")
axes[0, 0].grid(True, alpha=0.3)

# Plant species per year
axes[0, 1].plot(
    yearly_stats.index,
    yearly_stats["plant_species"],
    marker="o",
    linewidth=2,
    markersize=6,
    color="green",
)
axes[0, 1].set_title("Plant Species per Year")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Number of Plant Species")
axes[0, 1].grid(True, alpha=0.3)

# Pollinator species per year
axes[1, 0].plot(
    yearly_stats.index,
    yearly_stats["pollinator_species"],
    marker="o",
    linewidth=2,
    markersize=6,
    color="orange",
)
axes[1, 0].set_title("Pollinator Species per Year")
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("Number of Pollinator Species")
axes[1, 0].grid(True, alpha=0.3)

# Total interactions per year
axes[1, 1].plot(
    yearly_stats.index,
    yearly_stats["total_interactions"],
    marker="o",
    linewidth=2,
    markersize=6,
    color="purple",
)
axes[1, 1].set_title("Total Interactions per Year")
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Total Interaction Strength")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Network metrics calculation for selected networks
print("\n" + "=" * 80)
print("NETWORK METRICS ANALYSIS")
print("=" * 80)


def calculate_network_metrics(interactions_df, network_id):
    """Calculate comprehensive network metrics for a single network."""

    # Filter interactions for this network
    network_data = interactions_df[interactions_df["Network_id_full"] == network_id]

    if len(network_data) < 2:
        return None

    # Create network
    G = nx.Graph()

    # Add edges with weights
    for _, row in network_data.iterrows():
        G.add_edge(
            row["Plant_accepted_name"],
            row["Pollinator_accepted_name"],
            weight=row["Interaction"],
        )

    # Basic metrics
    metrics = {
        "network_id": network_id,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_shortest_path": (
            nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan
        ),
        "diameter": nx.diameter(G) if nx.is_connected(G) else np.nan,
        "connected_components": nx.number_connected_components(G),
        "largest_cc_size": (
            len(max(nx.connected_components(G), key=len))
            if nx.connected_components(G)
            else 0
        ),
        "plant_species": network_data["Plant_accepted_name"].nunique(),
        "pollinator_species": network_data["Pollinator_accepted_name"].nunique(),
        "total_strength": network_data["Interaction"].sum(),
        "mean_strength": network_data["Interaction"].mean(),
        "max_strength": network_data["Interaction"].max(),
        "min_strength": network_data["Interaction"].min(),
    }

    # Degree distribution
    degrees = [d for n, d in G.degree()]
    metrics["avg_degree"] = np.mean(degrees)
    metrics["max_degree"] = np.max(degrees)
    metrics["min_degree"] = np.min(degrees)

    # Weighted metrics
    weighted_degrees = [d for n, d in G.degree(weight="weight")]
    metrics["avg_weighted_degree"] = np.mean(weighted_degrees)
    metrics["max_weighted_degree"] = np.max(weighted_degrees)

    return metrics


# Calculate metrics for a sample of networks (to avoid memory issues)
sample_networks = network_stats.sample(min(100, len(network_stats)), random_state=42)[
    "Network_id_full"
].tolist()

print(f"Calculating network metrics for {len(sample_networks)} sample networks...")

network_metrics = []
for network_id in tqdm(sample_networks, desc="Calculating metrics"):
    metrics = calculate_network_metrics(interactions, network_id)
    if metrics:
        network_metrics.append(metrics)

network_metrics_df = pd.DataFrame(network_metrics)
print(f"Calculated metrics for {len(network_metrics_df)} networks")

# %%
# Network metrics visualization
print("\nNetwork metrics summary:")
print(network_metrics_df.describe())

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Network Metrics Distribution", fontsize=16, fontweight="bold")

# Density
axes[0, 0].hist(
    network_metrics_df["density"],
    bins=20,
    alpha=0.7,
    color="skyblue",
    edgecolor="black",
)
axes[0, 0].set_title("Network Density")
axes[0, 0].set_xlabel("Density")
axes[0, 0].set_ylabel("Number of Networks")

# Clustering coefficient
axes[0, 1].hist(
    network_metrics_df["avg_clustering"],
    bins=20,
    alpha=0.7,
    color="lightgreen",
    edgecolor="black",
)
axes[0, 1].set_title("Average Clustering Coefficient")
axes[0, 1].set_xlabel("Clustering Coefficient")
axes[0, 1].set_ylabel("Number of Networks")

# Average degree
axes[0, 2].hist(
    network_metrics_df["avg_degree"],
    bins=20,
    alpha=0.7,
    color="orange",
    edgecolor="black",
)
axes[0, 2].set_title("Average Degree")
axes[0, 2].set_xlabel("Average Degree")
axes[0, 2].set_ylabel("Number of Networks")

# Connected components
axes[1, 0].hist(
    network_metrics_df["connected_components"],
    bins=20,
    alpha=0.7,
    color="purple",
    edgecolor="black",
)
axes[1, 0].set_title("Number of Connected Components")
axes[1, 0].set_xlabel("Connected Components")
axes[1, 0].set_ylabel("Number of Networks")

# Largest connected component size
axes[1, 1].hist(
    network_metrics_df["largest_cc_size"],
    bins=20,
    alpha=0.7,
    color="red",
    edgecolor="black",
)
axes[1, 1].set_title("Largest Connected Component Size")
axes[1, 1].set_xlabel("Component Size")
axes[1, 1].set_ylabel("Number of Networks")

# Average shortest path (for connected networks)
connected_networks = network_metrics_df[network_metrics_df["avg_shortest_path"].notna()]
if len(connected_networks) > 0:
    axes[1, 2].hist(
        connected_networks["avg_shortest_path"],
        bins=20,
        alpha=0.7,
        color="brown",
        edgecolor="black",
    )
    axes[1, 2].set_title("Average Shortest Path Length")
    axes[1, 2].set_xlabel("Path Length")
    axes[1, 2].set_ylabel("Number of Networks")

plt.tight_layout()
plt.show()

# %%
# Species-level analysis
print("\n" + "=" * 80)
print("SPECIES-LEVEL ANALYSIS")
print("=" * 80)

# Most common species
plant_freq = interactions["Plant_accepted_name"].value_counts()
pollinator_freq = interactions["Pollinator_accepted_name"].value_counts()

print("Top 10 most frequent plant species:")
print(plant_freq.head(10))

print("\nTop 10 most frequent pollinator species:")
print(pollinator_freq.head(10))

# %%
# Species frequency visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 12))
fig.suptitle("Species Frequency Distribution", fontsize=16, fontweight="bold")

# Plant species frequency
top_plants = plant_freq.head(20)
axes[0].barh(range(len(top_plants)), top_plants.values, color="lightgreen")
axes[0].set_yticks(range(len(top_plants)))
axes[0].set_yticklabels(top_plants.index)
axes[0].set_title("Top 20 Most Frequent Plant Species")
axes[0].set_xlabel("Number of Interactions")

# Pollinator species frequency
top_pollinators = pollinator_freq.head(20)
axes[1].barh(range(len(top_pollinators)), top_pollinators.values, color="orange")
axes[1].set_yticks(range(len(top_pollinators)))
axes[1].set_yticklabels(top_pollinators.index)
axes[1].set_title("Top 20 Most Frequent Pollinator Species")
axes[1].set_xlabel("Number of Interactions")

plt.tight_layout()
plt.show()

# %%
# Network visualization for selected networks
print("\n" + "=" * 80)
print("NETWORK VISUALIZATION")
print("=" * 80)


def visualize_network(interactions_df, network_id, max_nodes=50):
    """Create a network visualization for a specific network."""

    # Filter interactions for this network
    network_data = interactions_df[interactions_df["Network_id_full"] == network_id]

    if len(network_data) == 0:
        print(f"No data found for network {network_id}")
        return

    # Create network
    G = nx.Graph()

    # Add edges with weights
    for _, row in network_data.iterrows():
        G.add_edge(
            row["Plant_accepted_name"],
            row["Pollinator_accepted_name"],
            weight=row["Interaction"],
        )

    # If network is too large, sample nodes
    if G.number_of_nodes() > max_nodes:
        # Keep nodes with highest degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[
            :max_nodes
        ]
        top_node_names = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_names).copy()
        print(
            f"Network too large ({G.number_of_nodes()} nodes), showing top {max_nodes} nodes"
        )

    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Draw edges
    edges = list(G.edges())
    weights = [G[u][v]["weight"] for u, v in edges]
    # If weights is empty, fallback to width=1
    if weights:
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w / 10.0 for w in weights])
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)

    # Draw nodes
    plant_nodes = [
        node for node in G.nodes() if node in network_data["Plant_accepted_name"].values
    ]
    pollinator_nodes = [
        node
        for node in G.nodes()
        if node in network_data["Pollinator_accepted_name"].values
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=plant_nodes,
        node_color="lightgreen",
        node_size=300,
        alpha=0.8,
        label="Plants",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=pollinator_nodes,
        node_color="orange",
        node_size=300,
        alpha=0.8,
        label="Pollinators",
    )

    # Add labels for a subset of nodes
    if G.number_of_nodes() <= 20:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(
        f"Network: {network_id}\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
    )
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Visualize a few representative networks
# Select networks with most interactions, greatest number of pollinators × flowers, and biggest size × interactions
network_with_most_interactions = network_stats.loc[
    network_stats["interaction_count"].idxmax(), "Network_id_full"
]
network_with_largest_size = network_stats.loc[
    (network_stats["plant_species"] * network_stats["pollinator_species"]).idxmax(),
    "Network_id_full",
]
network_with_biggest_size_x_interactions = network_stats.loc[
    (
        network_stats["plant_species"]
        * network_stats["pollinator_species"]
        * network_stats["interaction_count"]
    ).idxmax(),
    "Network_id_full",
]

selected_networks_for_viz = [
    network_with_most_interactions,
    network_with_largest_size,
    network_with_biggest_size_x_interactions,
]

print(f"\nSelected networks for visualization:")
print(f"1. Network with most interactions: {network_with_most_interactions}")
print(
    f"   - Interactions: {network_stats.loc[network_stats['Network_id_full'] == network_with_most_interactions, 'interaction_count'].iloc[0]:,}"
)
print(
    f"   - Plant species: {network_stats.loc[network_stats['Network_id_full'] == network_with_most_interactions, 'plant_species'].iloc[0]}"
)
print(
    f"   - Pollinator species: {network_stats.loc[network_stats['Network_id_full'] == network_with_most_interactions, 'pollinator_species'].iloc[0]}"
)

print(
    f"\n2. Network with largest size (pollinators × flowers): {network_with_largest_size}"
)
print(
    f"   - Size: {network_stats.loc[network_stats['Network_id_full'] == network_with_largest_size, 'plant_species'].iloc[0]} × {network_stats.loc[network_stats['Network_id_full'] == network_with_largest_size, 'pollinator_species'].iloc[0]} = {(network_stats.loc[network_stats['Network_id_full'] == network_with_largest_size, 'plant_species'].iloc[0] * network_stats.loc[network_stats['Network_id_full'] == network_with_largest_size, 'pollinator_species'].iloc[0]):,}"
)
print(
    f"   - Interactions: {network_stats.loc[network_stats['Network_id_full'] == network_with_largest_size, 'interaction_count'].iloc[0]:,}"
)

print(
    f"\n3. Network with biggest size × interactions (pollinators × flowers × interactions): {network_with_biggest_size_x_interactions}"
)
size_x_interactions = (
    network_stats.loc[
        network_stats["Network_id_full"] == network_with_biggest_size_x_interactions,
        "plant_species",
    ].iloc[0]
    * network_stats.loc[
        network_stats["Network_id_full"] == network_with_biggest_size_x_interactions,
        "pollinator_species",
    ].iloc[0]
    * network_stats.loc[
        network_stats["Network_id_full"] == network_with_biggest_size_x_interactions,
        "interaction_count",
    ].iloc[0]
)
print(
    f"   - Size × Interactions: {network_stats.loc[network_stats['Network_id_full'] == network_with_biggest_size_x_interactions, 'plant_species'].iloc[0]} × {network_stats.loc[network_stats['Network_id_full'] == network_with_biggest_size_x_interactions, 'pollinator_species'].iloc[0]} × {network_stats.loc[network_stats['Network_id_full'] == network_with_biggest_size_x_interactions, 'interaction_count'].iloc[0]:,} = {size_x_interactions:,}"
)

for network_id in selected_networks_for_viz:
    print(f"\nVisualizing network: {network_id}")
    visualize_network(interactions, network_id)

# %%
# Interaction strength analysis
print("\n" + "=" * 80)
print("INTERACTION STRENGTH ANALYSIS")
print("=" * 80)

# Distribution of interaction strengths
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Interaction Strength Analysis", fontsize=16, fontweight="bold")

# Overall distribution
axes[0, 0].hist(
    interactions["Interaction"], bins=50, alpha=0.7, color="skyblue", edgecolor="black"
)
axes[0, 0].set_title("Distribution of Interaction Strengths")
axes[0, 0].set_xlabel("Interaction Strength")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_yscale("log")

# Log distribution
axes[0, 1].hist(
    np.log10(interactions["Interaction"] + 1),
    bins=50,
    alpha=0.7,
    color="lightgreen",
    edgecolor="black",
)
axes[0, 1].set_title("Distribution of Log(Interaction Strength + 1)")
axes[0, 1].set_xlabel("Log(Interaction Strength + 1)")
axes[0, 1].set_ylabel("Frequency")

# Interaction strength by network size
axes[1, 0].scatter(
    network_stats["plant_species"] * network_stats["pollinator_species"],
    network_stats["mean_strength"],
    alpha=0.6,
    color="purple",
)
axes[1, 0].set_title("Network Size vs Mean Interaction Strength")
axes[1, 0].set_xlabel("Plant Species × Pollinator Species")
axes[1, 0].set_ylabel("Mean Interaction Strength")
axes[1, 0].grid(True, alpha=0.3)

# Interaction strength by year
yearly_strength = interactions.groupby("Year")["Interaction"].mean()
axes[1, 1].plot(
    yearly_strength.index, yearly_strength.values, marker="o", linewidth=2, markersize=6
)
axes[1, 1].set_title("Mean Interaction Strength by Year")
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Mean Interaction Strength")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Taxonomic analysis
print("\n" + "=" * 80)
print("TAXONOMIC ANALYSIS")
print("=" * 80)

# Pollinator orders
pollinator_orders = interactions["Pollinator_order"].value_counts()
print("Pollinator orders distribution:")
print(pollinator_orders)

# Plant families
plant_families = interactions["Plant_family"].value_counts()
print("\nTop 20 plant families:")
print(plant_families.head(20))

# %%
# Taxonomic visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 12))
fig.suptitle("Taxonomic Distribution", fontsize=16, fontweight="bold")

# Pollinator orders
top_orders = pollinator_orders.head(10)
axes[0].barh(range(len(top_orders)), top_orders.values, color="orange")
axes[0].set_yticks(range(len(top_orders)))
axes[0].set_yticklabels(top_orders.index)
axes[0].set_title("Top 10 Pollinator Orders")
axes[0].set_xlabel("Number of Interactions")

# Plant families
top_families = plant_families.head(15)
axes[1].barh(range(len(top_families)), top_families.values, color="lightgreen")
axes[1].set_yticks(range(len(top_families)))
axes[1].set_yticklabels(top_families.index)
axes[1].set_title("Top 15 Plant Families")
axes[1].set_xlabel("Number of Interactions")

plt.tight_layout()
plt.show()

# %%
# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("Dataset Summary:")
print(f"Total interactions: {len(interactions):,}")
print(f"Total networks: {interactions['Network_id_full'].nunique():,}")
print(f"Total studies: {interactions['Study_id'].nunique():,}")
print(f"Countries: {interactions['Country'].nunique()}")
print(f"Years: {interactions['Year'].min()}-{interactions['Year'].max()}")
print(f"Plant species: {interactions['Plant_accepted_name'].nunique():,}")
print(f"Pollinator species: {interactions['Pollinator_accepted_name'].nunique():,}")
print(f"Plant families: {interactions['Plant_family'].nunique():,}")
print(f"Pollinator orders: {interactions['Pollinator_order'].nunique():,}")

print(f"\nNetwork Statistics (from sample):")
if len(network_metrics_df) > 0:
    print(f"Mean density: {network_metrics_df['density'].mean():.4f}")
    print(f"Mean clustering: {network_metrics_df['avg_clustering'].mean():.4f}")
    print(f"Mean degree: {network_metrics_df['avg_degree'].mean():.2f}")
    print(
        f"Connected networks: {network_metrics_df['connected_components'].value_counts().get(1, 0)}"
    )

print(f"\nInteraction Strength:")
print(f"Mean: {interactions['Interaction'].mean():.4f}")
print(f"Median: {interactions['Interaction'].median():.4f}")
print(f"Std: {interactions['Interaction'].std():.4f}")
print(f"Min: {interactions['Interaction'].min():.4f}")
print(f"Max: {interactions['Interaction'].max():.4f}")

# %%
