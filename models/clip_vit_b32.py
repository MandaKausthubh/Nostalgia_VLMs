import torch
from torch import nn
from torch.optim import AdamW
from typing import Any, Dict, Optional, Tuple

from .base_model import VisionBaseModel, QLoRAConfig

try:
    from transformers import CLIPModel
    HAVE_TX = True
except Exception:
    HAVE_TX = False


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class _ClipB32Head(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.classifier(image_embeds)


class CLIPVitB32Classifier(VisionBaseModel):
    """
    CLIP ViT-B/32 backbone (HuggingFace transformers) with a linear classification head.

    Expects batches with keys:
      - images: FloatTensor [B, 3, H, W], normalized (ideally CLIP mean/std)
      - labels: LongTensor [B]
    """

    def __init__(
        self,
        num_classes: int,
        device: Optional[str] = None,
        qlora: Optional[QLoRAConfig] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        freeze_backbone: bool = False,
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        super().__init__(device=device, qlora=qlora)

    # ----- Model -----
    def build_model(self) -> nn.Module:
        assert HAVE_TX, "transformers is required: pip install transformers"
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        embed_dim = int(clip.config.projection_dim)

        if self.freeze_backbone:
            for p in clip.parameters():
                p.requires_grad = False

        # pack into a single torch module for potential PEFT wrapping
        module = nn.Module()
        module.clip = clip
        module.head = _ClipB32Head(embed_dim, self.num_classes)
        return module

    # ----- Steps -----
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)
        return {"images": images, "labels": labels}

    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        assert self.model is not None
        pixel_values = batch["images"]
        # HF CLIP provides get_image_features returning pooled embeddings
        image_embeds = self.model.clip.get_image_features(pixel_values=pixel_values)
        logits = self.model.head(image_embeds)
        return {"logits": logits, "image_embeds": image_embeds}

    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        return nn.functional.cross_entropy(outputs["logits"], batch["labels"])

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        assert self.model is not None
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer, None

    # ----- Utils -----
    @staticmethod
    def clip_normalization() -> Tuple[list, list]:
        """Return recommended mean/std for CLIP vision model."""
        return CLIP_MEAN, CLIP_STD


