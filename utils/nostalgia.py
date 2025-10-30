import torch
import tqdm

from typing import List, Tuple
from torch.utils.data import DataLoader

from .lanczos import compute_hessian_subspace, build_projection


def nostalgia_train(
    self,
    tasks: List[Tuple[DataLoader, DataLoader]],  # (train_loader, val_loader) pairs
    num_eigenthings: int = 10,
    ema_decay: float = 0.1,
    epochs: int = 3,
    grad_accum_steps: int = 1,
    log_interval: int = 50,
    use_amp: bool = True,
):
    """
    Continual learning via Hessian-projected updates.
    Each task fine-tunes LoRA parameters in the null-space of previous tasks' curvature.
    """
    assert self.model is not None, "Model not built."
    device = self.device
    self.scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    optimizer, scheduler = self.configure_optimizers()
    
    Q_avg = None  # Running average Hessian subspace
    for task_idx, (train_loader, val_loader) in enumerate(tasks):
        print(f"\n===== Task {task_idx+1}/{len(tasks)} =====")
        
        # (1) Compute curvature subspace from previous tasks
        if task_idx > 0:
            print("Computing Hessian subspace from previous tasks...")
            def loss_fn(batch):
                outputs = self.forward_step(batch)
                return self.compute_loss(outputs, batch)
            _, Q_t = compute_hessian_subspace(self.model, train_loader, loss_fn, num_eigenthings, device)
            
            if Q_avg is None:
                Q_avg = Q_t
            else:
                # EMA update: Q_avg = (1 - ema_decay) * Q_avg + ema_decay * Q_t
                # re-orthogonalize to prevent drift
                Q_avg = torch.linalg.qr((1 - ema_decay) * Q_avg + ema_decay * Q_t)[0]
            
            project = build_projection(Q_avg)
        else:
            project = lambda g: g  # No projection on first task

        # (2) Train current task
        self.model.train()
        global_step = 0
        for epoch in range(epochs):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=f"Task {task_idx+1} | Epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)

            for step_idx, batch in pbar:
                global_step += 1
                batch = self.preprocess_batch(batch)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.forward_step(batch)
                        loss = self.compute_loss(outputs, batch) / grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.forward_step(batch)
                    loss = self.compute_loss(outputs, batch) / grad_accum_steps
                    loss.backward()

                # Project gradients (LoRA parameters only)
                grads = []
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grads.append(p.grad.view(-1))
                g_flat = torch.cat(grads)
                g_proj = project(g_flat)

                # Write projected grads back
                idx = 0
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        numel = p.numel()
                        p.grad.copy_(g_proj[idx:idx+numel].view_as(p))
                        idx += numel

                if (global_step % grad_accum_steps) == 0:
                    if self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

                if (global_step % log_interval) == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if val_loader is not None:
            val_loss, val_top1, val_top5 = self.evaluate(val_loader)
            print(f"[Task {task_idx+1}] val_loss={val_loss:.4f} top1={val_top1:.2f} top5={val_top5:.2f}")
