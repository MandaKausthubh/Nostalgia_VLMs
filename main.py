import torch
from data_utils.standard_vision import CIFAR100Dataset, OxfordPetsDataset, ImageNetDataset
from models.clip_vit_b32 import CLIPVitB32Classifier
from utils.nostalgia import nostalgia_train

def collate_remove_none(batch):
    """Simple collate that skips samples where any item is None."""
    batch = [b for b in batch if b["images"] is not None and b["labels"] is not None]
    return {
        "images": torch.stack([b["images"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }

def main():
    # You can modify these per your needs, including batch sizes/paths.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    # 1. Prepare datasets and dataloaders
    datasets = [
        CIFAR100Dataset(split="train", max_samples=500),      # subset for speed; adjust as needed
        OxfordPetsDataset(split="train", max_samples=500),
        # Assuming you have val and train splits for each
        # For ImageNet: ensure correct folder structure and set max_samples <1000 for speed
        ImageNetDataset(split="train", max_samples=500),
    ]
    val_datasets = [
        CIFAR100Dataset(split="val", max_samples=200, download=False),
        OxfordPetsDataset(split="val", max_samples=200, download=False),
        ImageNetDataset(split="val", max_samples=200),
    ]

    train_loaders = [d.get_dataloader(batch_size=batch_size, shuffle=True, collate_fn=collate_remove_none, num_workers=2) for d in datasets]
    val_loaders = [d.get_dataloader(batch_size=batch_size, shuffle=False, collate_fn=collate_remove_none, num_workers=2) for d in val_datasets]

    # 2. Prepare model
    num_classes = 100  # You might want to finetune for every task's #classes, for continual learning this is simplistic
    model = CLIPVitB32Classifier(
        num_classes=num_classes,
        device=device,
        qlora=None,  # Optional: pass QLoRAConfig if you want adapters
        freeze_backbone=False,
    )

    # 3. Attach nostalgia_train method to model (if not class-inherited)
    setattr(model, 'nostalgia_train', nostalgia_train.__get__(model))

    # 4. Package dataloaders as list of (train, val) pairs
    tasks = list(zip(train_loaders, val_loaders))

    # 5. Run continual learning
    model.nostalgia_train(
        tasks=tasks,
        num_eigenthings=10,
        ema_decay=0.1,
        epochs=2,               # Use more for real runs
        grad_accum_steps=1,
        log_interval=10,
        use_amp=True,
    )

if __name__ == "__main__":
    main()