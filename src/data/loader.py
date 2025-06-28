"""
Data loader for EuPPollNet (European Plant-Pollinator Networks) dataset.

This module provides functions to load and process the EuPPollNet dataset,
which contains plant-pollinator interaction data from across Europe.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EuPPollNetLoader:
    """
    Data loader for EuPPollNet dataset.

    The EuPPollNet dataset contains:
    - Interaction data: Long format with individual plant-pollinator interactions
    - Plant taxonomy: Taxonomic information for plant species
    - Pollinator taxonomy: Taxonomic information for pollinator species
    - Metadata: Study-level information about sampling methods, locations, etc.
    - Flower counts: Optional flower abundance data
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the EuPPollNet data loader.

        Args:
            data_dir: Directory containing the EuPPollNet data files
        """
        self.data_dir = Path(data_dir)
        self.interactions = None
        self.plants_taxonomy = None
        self.pollinators_taxonomy = None
        self.metadata = None
        self.flower_counts = None

    def load_interactions(self) -> pd.DataFrame:
        """
        Load the interaction data from the compressed CSV file.

        Returns:
            DataFrame with interaction data
        """
        file_path = self.data_dir / "Interaction_data.csv.gz"

        if not file_path.exists():
            raise FileNotFoundError(f"Interaction data file not found: {file_path}")

        logger.info(f"Loading interaction data from {file_path}")

        # Try different encodings for reading the compressed CSV file
        encodings = ["utf-8", "latin-1", "cp1252"]
        interactions: Optional[pd.DataFrame] = None

        for encoding in encodings:
            try:
                interactions = pd.read_csv(
                    file_path, compression="gzip", encoding=encoding
                )
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if interactions is None:
            raise ValueError(
                "Could not read the interaction data file with any encoding"
            )

        # Convert date column to datetime
        interactions["Date"] = pd.to_datetime(interactions["Date"])

        # Add year, month, day columns for easier analysis
        interactions["Year"] = interactions["Date"].dt.year
        interactions["Month"] = interactions["Date"].dt.month
        interactions["Day"] = interactions["Date"].dt.day

        # Create a unique network identifier
        interactions["Network_id_full"] = (
            interactions["Study_id"]
            + "_"
            + interactions["Network_id"]
            + "_"
            + interactions["Year"].astype(str)
        )

        self.interactions = interactions

        logger.info(
            f"Loaded {len(interactions)} interactions from {interactions['Network_id_full'].nunique()} networks"
        )
        logger.info(
            f"Date range: {interactions['Date'].min()} to {interactions['Date'].max()}"
        )
        logger.info(f"Countries: {interactions['Country'].nunique()}")

        return interactions

    def load_plant_taxonomy(self) -> pd.DataFrame:
        """
        Load plant taxonomy data.

        Returns:
            DataFrame with plant taxonomic information
        """
        file_path = self.data_dir / "Plant_taxonomy.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Plant taxonomy file not found: {file_path}")
        logger.info(f"Loading plant taxonomy from {file_path}")
        encodings = ["utf-8", "latin-1", "cp1252"]
        plants_taxonomy = None
        for encoding in encodings:
            try:
                plants_taxonomy = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        if plants_taxonomy is None:
            raise ValueError("Could not read the plant taxonomy file with any encoding")
        self.plants_taxonomy = plants_taxonomy
        logger.info(f"Loaded taxonomy for {len(plants_taxonomy)} plant taxa")
        return plants_taxonomy

    def load_pollinator_taxonomy(self) -> pd.DataFrame:
        """
        Load pollinator taxonomy data.

        Returns:
            DataFrame with pollinator taxonomic information
        """
        file_path = self.data_dir / "Pollinator_taxonomy.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Pollinator taxonomy file not found: {file_path}")
        logger.info(f"Loading pollinator taxonomy from {file_path}")
        encodings = ["utf-8", "latin-1", "cp1252"]
        pollinators_taxonomy = None
        for encoding in encodings:
            try:
                pollinators_taxonomy = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        if pollinators_taxonomy is None:
            raise ValueError(
                "Could not read the pollinator taxonomy file with any encoding"
            )
        self.pollinators_taxonomy = pollinators_taxonomy
        logger.info(f"Loaded taxonomy for {len(pollinators_taxonomy)} pollinator taxa")
        return pollinators_taxonomy

    def load_metadata(self) -> pd.DataFrame:
        """
        Load study metadata.

        Returns:
            DataFrame with study metadata
        """
        file_path = self.data_dir / "Metadata.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        logger.info(f"Loading metadata from {file_path}")

        metadata = pd.read_csv(file_path)
        self.metadata = metadata

        logger.info(f"Loaded metadata for {len(metadata)} studies")

        return metadata

    def load_flower_counts(self) -> Optional[pd.DataFrame]:
        """
        Load flower count data (optional).

        Returns:
            DataFrame with flower count data or None if file doesn't exist
        """
        file_path = self.data_dir / "Flower_counts.csv"
        if not file_path.exists():
            logger.warning(f"Flower counts file not found: {file_path}")
            return None
        logger.info(f"Loading flower counts from {file_path}")
        encodings = ["utf-8", "latin-1", "cp1252"]
        flower_counts = None
        for encoding in encodings:
            try:
                flower_counts = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        if flower_counts is None:
            raise ValueError("Could not read the flower counts file with any encoding")
        self.flower_counts = flower_counts
        logger.info(f"Loaded flower counts for {len(flower_counts)} records")
        return flower_counts

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all EuPPollNet data files.

        Returns:
            Dictionary containing all loaded data
        """
        data = {}

        # Load all data files
        data["interactions"] = self.load_interactions()
        data["plants_taxonomy"] = self.load_plant_taxonomy()
        data["pollinators_taxonomy"] = self.load_pollinator_taxonomy()
        data["metadata"] = self.load_metadata()

        # Flower counts are optional
        flower_counts = self.load_flower_counts()
        if flower_counts is not None:
            data["flower_counts"] = flower_counts

        return data

    def get_network_matrices(self, binary: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Convert long-format interaction data to network matrices.

        Args:
            binary: If True, create binary matrices (presence/absence)
                   If False, create quantitative matrices (interaction counts)

        Returns:
            Dictionary mapping network IDs to interaction matrices
        """
        if self.interactions is None:
            self.load_interactions()

        networks = {}

        for network_id in self.interactions["Network_id_full"].unique():
            # Get interactions for this network
            network_data = self.interactions[
                self.interactions["Network_id_full"] == network_id
            ]

            # Create interaction matrix
            if binary:
                # Binary matrix: presence/absence
                matrix = network_data.groupby(
                    ["Plant_accepted_name", "Pollinator_accepted_name"]
                ).size()
                matrix = (matrix > 0).astype(int)
            else:
                # Quantitative matrix: interaction counts
                matrix = network_data.groupby(
                    ["Plant_accepted_name", "Pollinator_accepted_name"]
                ).size()

            # Convert to wide format
            matrix = matrix.unstack(fill_value=0)

            networks[network_id] = matrix

        logger.info(f"Created {len(networks)} network matrices")

        return networks

    def get_species_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get metadata for all plant and pollinator species.

        Returns:
            Tuple of (plants_metadata, pollinators_metadata)
        """
        if self.plants_taxonomy is None:
            self.load_plant_taxonomy()

        if self.pollinators_taxonomy is None:
            self.load_pollinator_taxonomy()

        # Get unique species from interactions
        if self.interactions is None:
            self.load_interactions()

        unique_plants = self.interactions["Plant_accepted_name"].unique()
        unique_pollinators = self.interactions["Pollinator_accepted_name"].unique()

        # Filter taxonomy to only include species that appear in interactions
        plants_metadata = self.plants_taxonomy[
            self.plants_taxonomy["Accepted_name"].isin(unique_plants)
        ].copy()

        pollinators_metadata = self.pollinators_taxonomy[
            self.pollinators_taxonomy["Accepted_name"].isin(unique_pollinators)
        ].copy()

        logger.info(
            f"Found metadata for {len(plants_metadata)} plant species and {len(pollinators_metadata)} pollinator species"
        )

        return plants_metadata, pollinators_metadata

    def get_network_metadata(self) -> pd.DataFrame:
        """
        Get metadata for all networks.

        Returns:
            DataFrame with network-level metadata
        """
        if self.interactions is None:
            self.load_interactions()

        if self.metadata is None:
            self.load_metadata()

        # Aggregate interaction data to network level
        network_metadata = (
            self.interactions.groupby("Network_id_full")
            .agg(
                {
                    "Study_id": "first",
                    "Network_id": "first",
                    "Sampling_method": "first",
                    "Authors_habitat": "first",
                    "EuPPollNet_habitat": "first",
                    "Bioregion": "first",
                    "Country": "first",
                    "Locality": "first",
                    "Latitude": "first",
                    "Longitude": "first",
                    "Year": "first",
                    "Month": ["min", "max"],
                    "Plant_accepted_name": "nunique",
                    "Pollinator_accepted_name": "nunique",
                    "Interaction": "sum",
                }
            )
            .reset_index()
        )

        # Flatten column names
        network_metadata.columns = [
            "Network_id_full",
            "Study_id",
            "Network_id",
            "Sampling_method",
            "Authors_habitat",
            "EuPPollNet_habitat",
            "Bioregion",
            "Country",
            "Locality",
            "Latitude",
            "Longitude",
            "Year",
            "Month_min",
            "Month_max",
            "Plant_species_richness",
            "Pollinator_species_richness",
            "Total_interactions",
        ]

        # Merge with study metadata
        network_metadata = network_metadata.merge(
            self.metadata[["Study_id", "DOI", "Sampling_days", "Sampling_period"]],
            on="Study_id",
            how="left",
        )

        logger.info(f"Created metadata for {len(network_metadata)} networks")

        return network_metadata


def create_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Create sample data for development and testing.

    Returns:
        Dictionary with sample EuPPollNet data
    """
    logger.info("Creating sample EuPPollNet data for development")

    # Sample interactions
    sample_interactions = pd.DataFrame(
        {
            "Study_id": ["1_Bartomeus", "1_Bartomeus", "1_Bartomeus", "2_Petanidou"],
            "Network_id": ["BAT1CA", "BAT1CA", "BAT1CA", "PET1GR"],
            "Network_id_full": [
                "1_Bartomeus_BAT1CA_2005",
                "1_Bartomeus_BAT1CA_2005",
                "1_Bartomeus_BAT1CA_2005",
                "2_Petanidou_PET1GR_2015",
            ],
            "Sampling_method": [
                "Focal_observation",
                "Focal_observation",
                "Focal_observation",
                "Random_census",
            ],
            "Authors_habitat": [
                "Coastal shrubland",
                "Coastal shrubland",
                "Coastal shrubland",
                "Mediterranean scrub",
            ],
            "EuPPollNet_habitat": [
                "Sclerophyllous vegetation",
                "Sclerophyllous vegetation",
                "Sclerophyllous vegetation",
                "Sclerophyllous vegetation",
            ],
            "Bioregion": [
                "Mediterranean",
                "Mediterranean",
                "Mediterranean",
                "Mediterranean",
            ],
            "Country": ["Spain", "Spain", "Spain", "Greece"],
            "Locality": [
                "Cap de Creus Natural Park",
                "Cap de Creus Natural Park",
                "Cap de Creus Natural Park",
                "Lesvos Island",
            ],
            "Latitude": [42.352, 42.352, 42.352, 39.2],
            "Longitude": [3.177, 3.177, 3.177, 26.1],
            "Date": ["2005-03-28", "2005-04-09", "2005-04-09", "2015-04-15"],
            "Year": [2005, 2005, 2005, 2015],
            "Month": [3, 4, 4, 4],
            "Day": [28, 9, 9, 15],
            "Interaction": [1, 1, 1, 1],
            "Plant_original_name": [
                "Urospermum picroides",
                "Sonchus tenerrinus",
                "Sonchus tenerrinus",
                "Cistus creticus",
            ],
            "Plant_accepted_name": [
                "Urospermum picroides",
                "Sonchus tenerrimus",
                "Sonchus tenerrimus",
                "Cistus creticus",
            ],
            "Plant_rank": ["SPECIES", "SPECIES", "SPECIES", "SPECIES"],
            "Plant_order": ["Asterales", "Asterales", "Asterales", "Malvales"],
            "Plant_family": ["Asteraceae", "Asteraceae", "Asteraceae", "Cistaceae"],
            "Plant_genus": ["Urospermum", "Sonchus", "Sonchus", "Cistus"],
            "Pollinator_original_name": [
                "Oedemera flavipes",
                "Anthidium sticticum",
                "Oedemera lurida",
                "Bombus terrestris",
            ],
            "Pollinator_accepted_name": [
                "Oedemera flavipes",
                "Rhodanthidium sticticum",
                "Oedemera lurida",
                "Bombus terrestris",
            ],
            "Pollinator_rank": ["SPECIES", "SPECIES", "SPECIES", "SPECIES"],
            "Pollinator_order": [
                "Coleoptera",
                "Hymenoptera",
                "Coleoptera",
                "Hymenoptera",
            ],
            "Pollinator_family": [
                "Oedemeridae",
                "Megachilidae",
                "Oedemeridae",
                "Apidae",
            ],
            "Pollinator_genus": ["Oedemera", "Rhodanthidium", "Oedemera", "Bombus"],
        }
    )

    # Sample plant taxonomy
    sample_plants_taxonomy = pd.DataFrame(
        {
            "Fixed_name": [
                "Urospermum picroides",
                "Sonchus tenerrimus",
                "Cistus creticus",
            ],
            "Rank": ["SPECIES", "SPECIES", "SPECIES"],
            "Status": ["ACCEPTED", "ACCEPTED", "ACCEPTED"],
            "Matchtype": ["EXACT", "EXACT", "EXACT"],
            "Scientific_name": [
                "Urospermum picroides (L.) F.W.Schmidt",
                "Sonchus tenerrimus L.",
                "Cistus creticus L.",
            ],
            "Canonical_name": [
                "Urospermum picroides",
                "Sonchus tenerrimus",
                "Cistus creticus",
            ],
            "Accepted_name": [
                "Urospermum picroides",
                "Sonchus tenerrimus",
                "Cistus creticus",
            ],
            "Phylum": ["Tracheophyta", "Tracheophyta", "Tracheophyta"],
            "Order": ["Asterales", "Asterales", "Malvales"],
            "Family": ["Asteraceae", "Asteraceae", "Cistaceae"],
            "Genus": ["Urospermum", "Sonchus", "Cistus"],
        }
    )

    # Sample pollinator taxonomy
    sample_pollinators_taxonomy = pd.DataFrame(
        {
            "Fixed_name": [
                "Oedemera flavipes",
                "Anthidium sticticum",
                "Oedemera lurida",
                "Bombus terrestris",
            ],
            "Rank": ["SPECIES", "SPECIES", "SPECIES", "SPECIES"],
            "Status": ["ACCEPTED", "SYNONYM", "ACCEPTED", "ACCEPTED"],
            "Matchtype": ["EXACT", "EXACT", "EXACT", "EXACT"],
            "Scientific_name": [
                "Oedemera flavipes (Fabricius, 1792)",
                "Anthidium sticticum (Fabricius, 1787)",
                "Oedemera lurida (Marsham, 1802)",
                "Bombus terrestris (Linnaeus, 1758)",
            ],
            "Canonical_name": [
                "Oedemera flavipes",
                "Anthidium sticticum",
                "Oedemera lurida",
                "Bombus terrestris",
            ],
            "Accepted_name": [
                "Oedemera flavipes",
                "Rhodanthidium sticticum",
                "Oedemera lurida",
                "Bombus terrestris",
            ],
            "Phylum": ["Arthropoda", "Arthropoda", "Arthropoda", "Arthropoda"],
            "Order": ["Coleoptera", "Hymenoptera", "Coleoptera", "Hymenoptera"],
            "Family": ["Oedemeridae", "Megachilidae", "Oedemeridae", "Apidae"],
            "Genus": ["Oedemera", "Rhodanthidium", "Oedemera", "Bombus"],
        }
    )

    # Sample metadata
    sample_metadata = pd.DataFrame(
        {
            "Study_number": [1, 2],
            "Study_id": ["1_Bartomeus", "2_Petanidou"],
            "DOI": ["https://doi.org/10.1007/s00442-007-0946-1", "NA"],
            "Year": [2005, 2015],
            "Sampling_method": ["Focal_observation", "Random_census"],
            "Min_date": ["2005-02-28", "2015-04-04"],
            "Max_date": ["2005-05-30", "2015-06-28"],
            "Sampling_days": [37, 18],
            "Sampling_period": [92, 86],
            "Total_interactions": [1480, 6183],
            "Country": ["Spain", "Greece"],
        }
    )

    return {
        "interactions": sample_interactions,
        "plants_taxonomy": sample_plants_taxonomy,
        "pollinators_taxonomy": sample_pollinators_taxonomy,
        "metadata": sample_metadata,
    }
