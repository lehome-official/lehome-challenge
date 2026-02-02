import os
from typing import Dict
from omegaconf import OmegaConf, DictConfig


class ChallengeGarmentLoader:
    """Loader class for loading garment configuration files from Challenge_Garment directory.

    This class is responsible for loading garment configurations from the specified directory
    structure, supporting both Release and Holdout versions, as well as different types of
    garments (long-sleeve top, short-sleeve top, long pants, short pants).
    """

    def __init__(self, base_path: str = "Assets/objects/Challenge_Garment"):
        """Initialize ChallengeGarmentLoader.

        Args:
            base_path (str): Base path to the Challenge_Garment directory,
                defaults to "Assets/objects/Challenge_Garment"
        """
        self.base_path = os.path.join(os.getcwd(), base_path)
        """Base path pointing to the Challenge_Garment directory"""
        self.garment_type_map: Dict[str, str] = {
            "Top_Long": "Top_Long",
            "Top_Short": "Top_Short",
            "Pant_Short": "Pant_Short",
            "Pant_Long": "Pant_Long",
        }
        """Garment type mapping dictionary that maps simplified type names to actual directory names"""

    def load_garment_config(
        self, garment_name: str, version: str = "Release"
    ) -> DictConfig:
        """Load configuration file for the specified garment.

        Based on the garment name and version, searches for and loads the JSON configuration
        file from the corresponding directory, returning an OmegaConf DictConfig object.

        Args:
            garment_name (str): Garment name in the format "Type_Length_Seen/Unseen_Index",
                e.g., "Top_Long_Unseen_0", "Top_Short_Seen_1", etc.
            version (str): Version type, either "Release" or "Holdout", defaults to "Release"

        Returns:
            DictConfig: OmegaConf configuration object containing the following fields:
                - id (int): Garment ID
                - asset_path (str): USD asset file path
                - visual_usd_paths (List[str]): List of visual material USD file paths
                - scale (List[float]): Scale factors in the format [x, y, z]
                - check_point (List[int], optional): List of checkpoints

        Raises:
            FileNotFoundError: Raised when the specified garment directory does not exist
            ValueError: Raised when no JSON configuration file is found in the directory
        """
        garment_type = self._get_garment_type(garment_name)
        garment_dir = os.path.join(self.base_path, version, garment_type, garment_name)

        if not os.path.isdir(garment_dir):
            raise FileNotFoundError(f"Garment directory not found: {garment_dir}")

        # Search for JSON configuration file in the directory (only one JSON file is allowed)
        filepath = None
        for filename in os.listdir(garment_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(garment_dir, filename)
                break

        if filepath is None:
            raise ValueError(
                f"No JSON configuration file found in directory: {garment_dir}"
            )

        # Load configuration file using OmegaConf
        return OmegaConf.load(filepath)

    def get_garment_type(self, garment_name: str) -> str:
        """
        Given a garment name (e.g., "Top_Long_Unseen_0"), return its canonical type string:
        One of ["top-long-sleeve", "top-short-sleeve", "short-pant", "long-pant"].

        This first uses _get_garment_type to obtain the mapped garment directory name,
        and then converts it to the desired canonical category.

        Args:
            garment_name (str): Name of the garment (e.g., "Top_Long_Unseen_0")

        Returns:
            str: One of "top-long-sleeve", "top-short-sleeve", "short-pant", "long-pant"
        """
        mapped_type = self._get_garment_type(garment_name)  # e.g., "Top_Long" etc.
        if mapped_type == "Top_Long":
            return "top-long-sleeve"
        elif mapped_type == "Top_Short":
            return "top-short-sleeve"
        elif mapped_type == "Pant_Short":
            return "short-pant"
        elif mapped_type == "Pant_Long":
            return "long-pant"
        else:
            raise ValueError(
                f"Unknown mapped garment type '{mapped_type}' for garment name '{garment_name}'."
            )

    def _get_garment_type(self, garment_name: str) -> str:
        """Extract and map garment type from garment name.

        Parses the first two parts of the garment name (e.g., "Top_Long") and maps it to
        the actual directory name (e.g., "Top_Long").

        Args:
            garment_name (str): Garment name in the format "Type_Length_Seen/Unseen_Index"

        Returns:
            str: Mapped garment type directory name, e.g., "Top_Long", "Pant_Short", etc.

        Raises:
            ValueError: Raised when the garment name format is invalid or the type is unknown
        """
        parts = garment_name.split("_")
        if len(parts) >= 2:
            garment_key = f"{parts[0]}_{parts[1]}"
        else:
            raise ValueError(
                f"Invalid garment name format: {garment_name}. "
                f"Expected format: 'Type_Length_Seen/Unseen_Index'"
            )

        if garment_key in self.garment_type_map:
            return self.garment_type_map[garment_key]
        else:
            raise ValueError(
                f"Unknown garment type: {garment_key}. "
                f"Valid types: {list(self.garment_type_map.keys())}"
            )


if __name__ == "__main__":
    # test the ChallengeGarmentLoader
    loader = ChallengeGarmentLoader(base_path="Assets/objects/Challenge_Garment")
    config = loader.load_garment_config("Top_Long_Unseen_0")
    print(config)
    config = loader.load_garment_config("Top_Short_Seen_1")
    print(config)
    config = loader.load_garment_config("Pant_Long_Unseen_0")
    print(config)
    config = loader.load_garment_config("Pant_Short_Seen_1")
    print(config)
