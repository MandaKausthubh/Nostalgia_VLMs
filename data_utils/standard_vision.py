import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .base_dataset import BaseDataset


def _default_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class _VisionClassificationWrapper(Dataset):
    """Wraps a torchvision classification dataset to always return a dict."""
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if isinstance(sample, Tuple) or isinstance(sample, list):
            image, label = sample[0], sample[1]
        elif isinstance(sample, Dict):
            image, label = sample.get("image"), sample.get("label") or sample.get("target")
        else:
            image, label = sample
        return {"image": image, "label": label}


class CIFAR100Dataset(BaseDataset):
    def __init__(self, split: str = "train", max_samples: Optional[int] = None, root: str = "~/.datasets/cifar100", image_size: int = 224, download: bool = True):
        super().__init__("CIFAR100", split, max_samples)
        self.root = os.path.expanduser(root)
        self.image_size = image_size
        self.download = download

    def load(self) -> Dataset:
        is_train = self.split == "train"
        tfm = _default_transforms(self.image_size)
        base = datasets.CIFAR100(root=self.root, train=is_train, transform=tfm, download=self.download)
        if self.max_samples:
            base.data = base.data[: self.max_samples]
            base.targets = base.targets[: self.max_samples]
        return _VisionClassificationWrapper(base)

    def format_sample(self, example) -> Dict[str, Any]:
        image = example["image"]  # tensor CxHxW
        label = example["label"]
        return {
            "images": image,   # tensor
            "text": None,      # not used for classification
            "labels": torch.tensor(label, dtype=torch.long),
        }


class OxfordPetsDataset(BaseDataset):
    def __init__(self, split: str = "train", max_samples: Optional[int] = None, root: str = "~/.datasets/oxford_pets", image_size: int = 224, download: bool = True):
        super().__init__("OxfordPets", split, max_samples)
        self.root = os.path.expanduser(root)
        self.image_size = image_size
        self.download = download

    def load(self) -> Dataset:
        is_train = self.split == "train"
        tfm = _default_transforms(self.image_size)
        base = datasets.OxfordIIITPet(root=self.root, split="trainval" if is_train else "test", target_types="category", transform=tfm, download=self.download)
        if self.max_samples:
            base._images = base._images[: self.max_samples]
            base._labels = base._labels[: self.max_samples]
        return _VisionClassificationWrapper(base)

    def format_sample(self, example) -> Dict[str, Any]:
        image = example["image"]
        label = example["label"]
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        return {"images": image, "text": None, "labels": label}


class ImageNetDataset(BaseDataset):
    """
    ImageNet dataset wrapper. Note: torchvision cannot auto-download ImageNet due to licensing.
    Provide the extracted ImageNet directory in `root` with subfolders `train/` and `val/`.
    """
    def __init__(self, split: str = "train", max_samples: Optional[int] = None, root: str = "~/datasets/imagenet", image_size: int = 224):
        super().__init__("ImageNet", split, max_samples)
        self.root = os.path.expanduser(root)
        self.image_size = image_size

    def load(self) -> Dataset:
        split_dir = "train" if self.split == "train" else "val"
        tfm = _default_transforms(self.image_size)
        base = datasets.ImageFolder(root=os.path.join(self.root, split_dir), transform=tfm)
        if self.max_samples:
            # ImageFolder uses samples list
            base.samples = base.samples[: self.max_samples]
            base.targets = base.targets[: self.max_samples]
        return _VisionClassificationWrapper(base)

    def format_sample(self, example) -> Dict[str, Any]:
        image = example["image"]
        label = example["label"]
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        return {"images": image, "text": None, "labels": label}


