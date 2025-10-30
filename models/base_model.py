from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import os
import warnings

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


# Optional deps: PEFT (LoRA) + bitsandbytes (k-bit) for QLoRA
try:
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    HAVE_PEFT = True
except Exception:
    HAVE_PEFT = False

try:
    import bitsandbytes as bnb  # noqa: F401
    from transformers import BitsAndBytesConfig
    from transformers import AutoConfig  # type: ignore
    from transformers.utils import is_torch_bf16_gpu_available
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False


@dataclass
class QLoRAConfig:
    """
    Configuration for enabling QLoRA adapters on a backbone model.
    - If provided, we will quantize the base model to 4-bit/8-bit (bitsandbytes)
      and attach LoRA adapters (PEFT) to selected target modules.
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    target_modules: Optional[List[str]] = None  # e.g., ["q_proj", "v_proj"]
    task_type: str = "FEATURE_EXTRACTION"  # For vision/backbone-only; e.g., "SEQ_CLS" for cls
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_compute_dtype: Optional[str] = None  # "float16" | "bfloat16" | "float32"
    bnb_4bit_use_double_quant: bool = True


class VisionBaseModel:
    """
    Vision-focused base model with optional QLoRA adapters.

    Subclass must implement:
      - build_model() -> nn.Module
      - preprocess_batch(batch) -> Dict[str, Any]
      - forward_step(batch) -> Dict[str, torch.Tensor]
      - compute_loss(outputs, batch) -> torch.Tensor
      - configure_optimizers() -> (optimizer, scheduler|None)
    """

    def __init__(self, device: Optional[str] = None, qlora: Optional[QLoRAConfig] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.qlora_cfg = qlora

        backbone = self.build_model()
        if qlora is not None:
            backbone = self._maybe_enable_qlora(backbone, qlora)
        self.model = backbone.to(self.device)

    # -------- Abstract API (override in subclass) --------
    @abstractmethod
    def build_model(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self) -> Tuple[Optimizer, Optional[Any]]:
        raise NotImplementedError

    # -------- QLoRA helpers --------
    def _maybe_enable_qlora(self, model: nn.Module, cfg: QLoRAConfig) -> nn.Module:
        if not HAVE_PEFT:
            warnings.warn("PEFT not installed; skipping QLoRA.")
            return model
        if not (cfg.use_4bit or cfg.use_8bit):
            # LoRA only without quantization
            return self._attach_lora(model, cfg)

        if not HAVE_BNB:
            warnings.warn("bitsandbytes/transformers not installed; cannot quantize model. Proceeding with LoRA only.")
            return self._attach_lora(model, cfg)

        # Configure 4-bit/8-bit quantization
        bnb_cfg = None
        if cfg.use_4bit:
            compute_dtype = torch.float16
            if cfg.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_available():
                if "is_torch_bf16_gpu_available" in globals() and is_torch_bf16_gpu_available():
                    compute_dtype = torch.bfloat16
            elif cfg.bnb_4bit_compute_dtype == "float32":
                compute_dtype = torch.float32

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif cfg.use_8bit:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

        # If using HF transformers models, some accept quantization config via from_pretrained; for generic nn.Module
        # we attempt to move to half precision where safe. Users should override build_model to return a quant-ready module.
        if bnb_cfg is not None:
            # Heuristic: convert model weights to half for memory if device supports
            try:
                if self.device.type == "cuda":
                    model = model.half()
            except Exception:
                pass

        return self._attach_lora(model, cfg)

    def _attach_lora(self, model: nn.Module, cfg: QLoRAConfig) -> nn.Module:
        target_modules = cfg.target_modules
        if target_modules is None:
            # Best-effort default: common attention proj names used in ViT/transformers backbones
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

        # Map task type string to PEFT enum if available
        task_enum = getattr(TaskType, cfg.task_type, TaskType.FEATURE_EXTRACTION) if HAVE_PEFT else None
        lora_cfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.bias,
            target_modules=target_modules,
            task_type=task_enum,
        )
        peft_model = get_peft_model(model, lora_cfg)
        peft_model.print_trainable_parameters()
        return peft_model

    # -------- IO --------
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self.model is None:
            return
        # If model is PEFT-wrapped, save adapters; else save full state
        if HAVE_PEFT and hasattr(self.model, "save_pretrained"):
            try:
                self.model.save_pretrained(save_dir)
                return
            except Exception:
                pass
        torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    def load(self, load_dir: str):
        if self.model is None:
            return
        # Try PEFT adapter load first
        if HAVE_PEFT and hasattr(self.model, "load_adapter"):
            try:
                self.model.load_adapter(load_dir, adapter_name="default")
                self.model.set_adapter("default")
                return
            except Exception:
                pass
        # Fallback to full-state dict
        state = torch.load(os.path.join(load_dir, "pytorch_model.bin"), map_location=self.device)
        self.model.load_state_dict(state, strict=False)


    def unload_and_merge(self):
        """
        Merges LoRA/PEFT weights (if any) back into the original model,
        removes adapter layers, and sets the model back to the merged state.
        If not a PEFT model, does nothing.
        """
        if self.model is None:
            warnings.warn("No model to unload and merge.")
            return

        # Only do this if PEFT is available and model is from get_peft_model
        if HAVE_PEFT and hasattr(self.model, "merge_and_unload"):
            try:
                self.model = self.model.merge_and_unload()
                # Optionally: print to confirm operation
                print("Merged adapters (if present) back into the base model and removed PEFT layers.")
            except Exception as e:
                warnings.warn(f"Failed to merge/unload PEFT adapters: {e}")
        else:
            warnings.warn("Model is not a PEFT-wrapped model or PEFT is not installed; nothing to merge.")


    # -------- Training / Evaluation --------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        log_interval: int = 50,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        max_grad_norm: Optional[float] = 1.0,
    ) -> None:
        assert self.model is not None, "Model not built"
        optimizer, scheduler = self.configure_optimizers()

        self.scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

        self.model.train()
        global_step = 0
        for epoch in range(epochs):
            running_loss = 0.0
            running_top1 = 0.0
            running_top5 = 0.0
            running_count = 0

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=f"Epoch {epoch+1}/{epochs}")
            optimizer.zero_grad(set_to_none=True)

            for step_idx, batch in pbar:
                global_step += 1
                inputs = self.preprocess_batch(batch)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.forward_step(inputs)
                        loss = self.compute_loss(outputs, inputs) / grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.forward_step(inputs)
                    loss = self.compute_loss(outputs, inputs) / grad_accum_steps
                    loss.backward()

                running_loss += float(loss.item())

                # Optional: classification metrics if logits+labels are present
                logits = outputs.get("logits")
                labels = inputs.get("labels")
                if logits is not None and labels is not None and logits.ndim == 2:
                    top1, top5 = self._batch_topk(logits.detach(), labels.detach())
                    running_top1 += top1
                    running_top5 += top5
                    running_count += labels.size(0)

                if (global_step % grad_accum_steps) == 0:
                    if self.scaler is not None:
                        if max_grad_norm is not None:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        try:
                            scheduler.step()
                        except Exception:
                            pass

                if (global_step % log_interval) == 0:
                    avg_loss = running_loss * grad_accum_steps / log_interval
                    if running_count > 0:
                        top1_acc = 100.0 * running_top1 / running_count
                        top5_acc = 100.0 * running_top5 / running_count
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "top1": f"{top1_acc:.2f}", "top5": f"{top5_acc:.2f}"})
                    else:
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    running_loss = 0.0
                    running_top1 = 0.0
                    running_top5 = 0.0
                    running_count = 0

            if val_loader is not None:
                val_loss, val_top1, val_top5 = self.evaluate(val_loader)
                tqdm.write(f"[val] epoch={epoch+1} loss={val_loss:.4f} top1={val_top1:.2f} top5={val_top5:.2f}")

    @staticmethod
    def _batch_topk(logits: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, int] = (1, 5)) -> Tuple[float, float]:
        max_k = max(topk)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            res.append(correct_k)
        return float(res[0]), float(res[1])

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        assert self.model is not None, "Model not built"
        self.model.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_count = 0
        with torch.no_grad():
            for _, batch in enumerate(tqdm(data_loader, ncols=100, desc="Valid")):
                inputs = self.preprocess_batch(batch)
                outputs = self.forward_step(inputs)
                loss = self.compute_loss(outputs, inputs)
                total_loss += float(loss.item())

                logits = outputs.get("logits")
                labels = inputs.get("labels")
                if logits is not None and labels is not None and logits.ndim == 2:
                    top1, top5 = self._batch_topk(logits, labels)
                    total_top1 += top1
                    total_top5 += top5
                    total_count += labels.size(0)
        self.model.train()
        mean_loss = total_loss / max(1, len(data_loader))
        top1_acc = 100.0 * total_top1 / max(1, total_count)
        top5_acc = 100.0 * total_top5 / max(1, total_count)
        return mean_loss, top1_acc, top5_acc


