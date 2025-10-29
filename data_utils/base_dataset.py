from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator


class BaseDataset(ABC):
    """Generic dataset wrapper for continual learning tasks."""

    # Inner class to format dataset samples
    class FormatedDataset(Dataset):
        def __init__(self, base_dataset, formate_fn):
            self.base_dataset = base_dataset
            self.formate_fn = formate_fn

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            example = self.base_dataset[idx]
            return self.formate_fn(example)


    def __init__(self, name: str, split: str = "train", max_samples: Optional[int] = None):
        self.name = name
        self.split = split
        self.max_samples = max_samples
        self.dataset: Optional[Dataset] = None
        self.task_prompt = ""

    @abstractmethod
    def load(self) -> Dataset:
        """Load and preprocess dataset from HF/Pytorch. Ensure that it returns a pytorch Dataset."""
        pass

    @abstractmethod
    def format_sample(self, example) -> Dict[str, Any]:
        """Format raw example → VLM input (image, text, label)."""
        pass

    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:

        if self.dataset is None:
            self.dataset = self.load()

        formated_dataset = self.FormatedDataset(self.dataset, self.format_sample)

        return DataLoader(
            formated_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=default_data_collator,
            num_workers=num_workers,
            pin_memory=True
        )


