import os
import gdown
import tarfile
import requests
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import h5py
from .base_dataset import BaseDataset
from typing import Dict, Any


class VLUEDataset(BaseDataset):
    """
    VLUE (Vision-Language Understanding Evaluation) benchmark.
    Covers Image-Text Retrieval, Visual Grounding, Visual Reasoning, and VQA.

    Automatically downloads:
    - Annotations from Google Drive (JSON files with captions)
    - Image data from ERDA (HDF5 files)

    Extracts them, and returns a ready-to-use PyTorch Dataset.

    Args:
        split (str): Which split to load ("train", "val", "test").
        max_samples (int): Number of samples to keep (for debugging).
        root (str): Path to store downloaded dataset.
        drive_file_id (str): Public Google Drive File ID for the VLUE annotations.
        erda_base_url (str): Base URL for ERDA file sharing service.
        working_directory (str): Working directory identifier from ERDA share.
        language_codes (list): Language codes for MARVL HDF5 files.
    """

    def __init__(
        self,
        split="test",
        max_samples=None,
        root="datasets",
        gdrive_url="https://drive.google.com/uc?id=1XFz1Vtz7MCBLn4_1QEojhFJ5Iw3eH3X4",
        erda_base_url="https://sid.erda.dk/share_redirect/hmoEs4a3oG/marvl-id_boxes36.h5",
        working_directory="hmoEs4a3oG",
        language_codes=["id", "sw", "ta", "tr", "zh"]
    ):
        # https://sid.erda.dk/share_redirect/hmoEs4a3oG/marvl-id_boxes36.h5
        super().__init__("VLUE", split, max_samples)
        self.root = os.path.join(os.path.dirname(__file__), root)
        self.dataset_dir = os.path.join(self.root, "VLUE")
        self.dataset_file = os.path.join(self.dataset_dir, "finetune_gdrive.tar")
        self.erda_file = os.path.join(self.dataset_dir, "finetune_erda.h5")

        self.gdrive_url = gdrive_url
        self.erda_base_url = erda_base_url

        self._ensure_dataset_google_drive()
        self._ensure_dataset_erda()
        self.dataset = self.load()  # returns torch.utils.data.Dataset


    def _ensure_dataset_google_drive(self):
        
        print(f"Downloading dataset from Google Drive")
        g_drive_dataset_dir = os.path.join(self.dataset_dir, "annotations")
        if not os.path.exists(g_drive_dataset_dir):
            os.makedirs(g_drive_dataset_dir)
            print(f"Created directory {g_drive_dataset_dir}")
            gdown.download(self.gdrive_url, self.dataset_file, quiet=False)
            print(f"Downloaded dataset to {self.dataset_file}")
            with tarfile.open(self.dataset_file, 'r') as tar:
                tar.extractall(path=g_drive_dataset_dir)
            print(f"Extracted dataset to {g_drive_dataset_dir}")
            os.remove(self.dataset_file)
            print(f"Removed {self.dataset_file}")
        else:
            print(f"Dataset already exists in {g_drive_dataset_dir}")
    
    def _ensure_dataset_erda(self):
        print(f"Downloading dataset from ERDA")
        images_dir = os.path.join(self.dataset_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            print(f"Created directory {images_dir}")
            with requests.get(self.erda_base_url, stream=True) as response:
                response.raise_for_status()
                with open(self.erda_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded dataset to {self.erda_file}")

                # Extract from h5py files
                # with h5py.File(self.erda_file, "r") as f:
                #     print("Keys in HDF5 file:", list(f.keys()))

                print(f"Extracted dataset to {images_dir}")
                os.remove(self.erda_file)
                print(f"Removed {self.erda_file}")
        else:
            print(f"Dataset already exists in {images_dir}")
    
    def load(self):
        pass

    def format_sample(self, example) -> Dict[str, Any]:
        pass



