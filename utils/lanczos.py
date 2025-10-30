from hessian_eigenthings import compute_hessian_eigenthings
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_hessian_subspace(model, dataloader, loss_fn, num_eigenthings=10, device="cuda"):
    """
    Estimate top-k Hessian eigenpairs (λ_i, v_i) for LoRA trainable parameters.
    Returns eigenvectors stacked into a matrix Q.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    def closure(data_iter):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = loss_fn(batch)
        return loss

    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model=model,
        dataloader=dataloader,
        loss=closure,
        params=params,
        num_eigenthings=num_eigenthings,
        mode="lanczos",
        full_dataset=False,
        use_cuda=(device == "cuda"),
    )

    # Stack eigenvectors as columns
    Q = torch.stack([v.flatten() for v in eigenvecs], dim=1)
    return eigenvals, Q



def build_projection(Q: torch.Tensor):
    """Return a projection function g -> g_proj."""
    def project(g_flat: torch.Tensor):
        if Q.numel() == 0:
            return g_flat
        return g_flat - Q @ (Q.T @ g_flat)
    return project
