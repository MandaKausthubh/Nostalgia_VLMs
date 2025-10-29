import os
import tarfile
import zipfile
import requests
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from PIL import Image
import h5py
from .base_dataset import BaseDataset


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
        root="~/.datasets/data/VLUE",
        drive_file_id="1XFz1Vtz7MCBLn4_1QEojhFJ5Iw3eH3X4",
        erda_base_url="https://filesender.erda.dk",
        working_directory="hmoEs4a3oG",
        language_codes=["id", "sw", "ta", "tr", "zh"]
    ):
        super().__init__("VLUE", split, max_samples)
        self.root = os.path.expanduser(root)
        self.dataset_dir = os.path.join(self.root, "VLUE")
        self.drive_file_id = drive_file_id
        self.erda_base_url = erda_base_url
        self.working_directory = working_directory
        self.language_codes = language_codes
        self.archive_path = os.path.join(self.root, "vlue_dataset.tar")
        self.h5_files_dir = os.path.join(self.dataset_dir, "h5_files")

        self._ensure_dataset()
        self.dataset = self.load()  # returns torch.utils.data.Dataset

    def _ensure_dataset(self):
        """Download annotations from Google Drive and image data from ERDA."""
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.h5_files_dir, exist_ok=True)

        # Step 1: Download annotations from Google Drive
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.root, exist_ok=True)
            print("🔽 Downloading VLUE annotations from Google Drive...")
            url = f"https://drive.google.com/uc?id={self.drive_file_id}"
            self._download_with_progress(url, self.archive_path)
            print("📦 Extracting annotations...")
            self._extract_with_progress(self.archive_path, self.root)
            print("✅ Annotations downloaded and extracted successfully.")
        else:
            print("✅ Annotations found locally.")

        # Step 2: Download HDF5 image data files from ERDA
        print("🔽 Checking ERDA files for image data...")
        for lang_code in self.language_codes:
            filename = f"marvl-{lang_code}_boxes36.h5"
            h5_path = os.path.join(self.h5_files_dir, filename)

            if not os.path.exists(h5_path):
                print(f"🔽 Downloading {filename} from ERDA...")
                self._download_from_erda(filename, h5_path)
            else:
                print(f"✅ {filename} found locally.")

        print("✅ VLUE dataset files downloaded successfully.")

    def _download_with_progress(self, url, dest_path):
        """Download file with tqdm progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(dest_path, "wb") as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc="Downloading annotations",
            ncols=80
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))

    def _download_from_erda(self, filename, dest_path):
        """Download file from ERDA file sharing service with tqdm progress bar."""
        # Construct URL for ERDA file sharing service
        url = f"{self.erda_base_url}/d/{self.working_directory}/{filename}"

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(dest_path, "wb") as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=f"Downloading {filename}",
            ncols=80
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))

    def _extract_with_progress(self, archive_path, extract_to):
        """Extract .tar or .zip file with tqdm progress bar."""
        if archive_path.endswith((".tar", ".tar.gz")):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                members = tar_ref.getmembers()
                for member in tqdm(members, desc="Extracting", ncols=80):
                    tar_ref.extract(member, extract_to)
        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                members = zip_ref.namelist()
                for member in tqdm(members, desc="Extracting", ncols=80):
                    zip_ref.extract(member, extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

    def load(self):
        """
        Load and return a PyTorch Dataset combining VLUE annotations (JSON)
        and image data (HDF5 files).
        """
        # Load annotations from JSON files
        annotations_path = os.path.join(self.dataset_dir, "data")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Expected 'data/' folder under {self.dataset_dir}")

        annotation_files = [
            os.path.join(annotations_path, f)
            for f in os.listdir(annotations_path)
            if f.endswith(".json")
        ]

        all_samples = []
        for ann_path in annotation_files:
            with open(ann_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
                else:
                    all_samples.append(data)

        if self.max_samples:
            all_samples = all_samples[:self.max_samples]

        return VLUEPyTorchDataset(all_samples, self.dataset_dir, self.h5_files_dir)

    def format_sample(self, example):
        """Format a single example to VLM input format."""
        return example


class VLUEPyTorchDataset(Dataset):
    """
    PyTorch Dataset that combines VLUE annotations (from JSON)
    with image data (from HDF5 files).
    """
    def __init__(self, samples, annotations_dir, h5_files_dir):
        self.samples = samples
        self.annotations_dir = annotations_dir
        self.h5_files_dir = h5_files_dir
        self.h5_datasets = {}
        self._load_all_h5_datasets()

    def _load_all_h5_datasets(self):
        """Pre-load all HDF5 datasets into memory for efficient access."""
        h5_files = [
            os.path.join(self.h5_files_dir, f)
            for f in os.listdir(self.h5_files_dir)
            if f.endswith(".h5") and f.startswith("marvl-")
        ]

        for h5_path in h5_files:
            try:
                self.h5_datasets[h5_path] = h5py.File(h5_path, "r")
                keys = list(self.h5_datasets[h5_path].keys())
                print(f"Opened HDF5 file: {h5_path} with keys: {keys}")
            except Exception as e:
                print(f"Warning: Could not open {h5_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get text/caption from annotations
        text = sample.get("text", "") or sample.get("caption", "") or sample.get("label", "")
        label = sample.get("label", None)

        # Try to load from HDF5 files if image path isn't present
        image = self._load_image_from_h5(sample)
        image_features = self._get_image_features_from_h5(sample)

        return {
            "image": image,
            "text": text,
            "label": label,
            "image_features": image_features,
            "metadata": sample
        }

    def _load_image_from_h5(self, sample):
        """Try to load image from HDF5 files based on sample metadata.
        Adjust this to your HDF5 structure if you need raw images."""
        for _, h5_file in self.h5_datasets.items():
            if "images" in h5_file:
                try:
                    return None  # placeholder, depends on actual structure
                except Exception:
                    pass
        return None

    def _get_image_features_from_h5(self, sample):
        """Extract image features from HDF5 files based on sample ID or index."""
        for _, h5_file in self.h5_datasets.items():
            for key in ["image_features", "features", "img_feat"]:
                if key in h5_file:
                    try:
                        return h5_file[key]
                    except Exception:
                        pass
        return None

    def __del__(self):
        for h5_file in self.h5_datasets.values():
            if h5_file is not None:
                try:
                    h5_file.close()
                except Exception:
                    pass


