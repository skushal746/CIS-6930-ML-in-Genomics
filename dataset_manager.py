import os
import zipfile
import subprocess
from pathlib import Path

class DatasetManager:
    def __init__(self, username: str, key: str, competition: str = "msk-redefining-cancer-treatment"):
        self.username = username
        self.key = key
        self.competition = competition
        self.zip_name = f"{competition}.zip"
        self.target_dir = Path(competition)

        # Set environment variables for Kaggle API
        os.environ['KAGGLE_USERNAME'] = self.username
        os.environ['KAGGLE_KEY'] = self.key

    def download(self):
        """Downloads the dataset if the ZIP file doesn't already exist."""
        if os.path.exists(self.zip_name):
            print(f"--- ZIP file '{self.zip_name}' already exists. Skipping download. ---")
            return

        print(f"--- Starting download for competition: {self.competition} ---")
        try:
            # We use subprocess to call the CLI already installed in your venv
            subprocess.run(["./.venv/bin/kaggle", "competitions", "download", "-c", self.competition], check=True)
            print("--- Download complete! ---")
        except subprocess.CalledProcessError as e:
            print(f"Error during download: {e}")

    def extract(self):
        """Unzips the dataset and cleans up internal ZIP files if necessary."""
        if not os.path.exists(self.zip_name):
            print(f"Error: {self.zip_name} not found. Cannot unzip.")
            return

        print(f"--- Extracting {self.zip_name} to {self.target_dir}/ ---")
        with zipfile.ZipFile(self.zip_name, 'r') as zip_ref:
            zip_ref.extractall(self.target_dir)
        
        # This specific Kaggle dataset often contains sub-ZIP files (training_variants.zip, etc.)
        # We'll unzip any .zip files found inside the target directory
        for item in os.listdir(self.target_dir):
            if item.endswith(".zip"):
                item_path = self.target_dir / item
                print(f"--- Extracting sub-ZIP: {item} ---")
                with zipfile.ZipFile(item_path, 'r') as sub_zip:
                    sub_zip.extractall(self.target_dir)
                os.remove(item_path) # Clean up sub-zips

        print("--- Extraction and cleanup complete! ---")

if __name__ == "__main__":
    # Credentials
    USER = os.getenv("KAGGLE_USERNAME")
    KEY = os.getenv("KAGGLE_KEY")
    print(KEY)

    manager = DatasetManager(USER, KEY)
    manager.download()
    manager.extract()
