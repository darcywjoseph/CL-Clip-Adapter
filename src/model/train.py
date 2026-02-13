import torch
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Optional, Any

from model.model import Adapter
import logging

logger = logging.getLogger("experiment_logs")

def contrastive_loss(features: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:

    features = F.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T) / temperature

    labels = labels.unsqueeze(1)
    mask_positive = (labels == labels.T).float()
    mask_negative = (labels != labels.T).float()
    mask_positive.fill_diagonal_(0)

    exp_sim = torch.exp(similarity)
    positive_sum = (exp_sim * mask_positive).sum(dim=1)
    negative_sum = (exp_sim * mask_negative).sum(dim=1)

    eps = 1e-8
    invariant_loss = -torch.log((positive_sum + eps) / (positive_sum + negative_sum + eps))
    invariant_loss = invariant_loss.mean()

    return invariant_loss
    
@torch.no_grad()
def _snapshot_params(model: torch.nn.Module) -> dict[str, Tensor]:
    """Clone current trainable parameters into a dict."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

def compute_fisher_diagonal(
    model: torch.nn.Module,
    dataloader,
    device: str,
    fisher_sample_size: int = 100,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Compute diagonal Fisher information estimate for EWC.
    """
    model = model.to(device)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    mean = _snapshot_params(model)
    fisher = {n: torch.zeros_like(p, device=device) for n, p in params.items()}

    model.eval()
    count = 0

    for images, labels in dataloader:
        if count >= fisher_sample_size:
            break

        images = images.to(device)
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        _, logits = model(images)

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        for n, p in params.items():

            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

        count += images.size(0)

    if count == 0:
        raise ValueError("0")

    for n in fisher:
        fisher[n] /= count

    return mean, fisher


def ewc_penalty(
    model: torch.nn.Module,
    mean: dict[str, Tensor],
    fisher: dict[str, Tensor],
) -> torch.Tensor:
    """
    Compute EWC quadratic penalty.
    """
    loss = torch.zeros((), device=next(model.parameters()).device)

    for n, p in model.named_parameters():

        if p.requires_grad and n in fisher:
            loss = loss + (fisher[n] * (p - mean[n]) ** 2).sum()

    return loss


def train_single_epoch(
        loader: DataLoader, 
        model: Adapter, 
        optimizer: Optimizer, 
        use_contrastive: bool =True, 
        w:float=0.5, 
        device: str="cuda",
        ewc_mean: Optional[dict[str, Tensor]] = None,
        ewc_fisher: Optional[dict[str, Tensor]] = None,
        ewc_lambda: float = 0.0,
    ) -> float:

    model.train()
    total_loss = 0
    total_samples = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        features, logits = model(images)

        classif_loss = F.cross_entropy(logits, labels)

        if use_contrastive:
            contr_loss = contrastive_loss(features, labels)
            loss = classif_loss + w * contr_loss
        else:
            loss = classif_loss

        if ewc_mean is not None and ewc_fisher is not None and ewc_lambda > 0.0:
            loss = loss + ewc_lambda * ewc_penalty(model, ewc_mean, ewc_fisher)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples
    

def build_optimizer(cfg: Any, model: Adapter) -> optim.Optimizer:
    params = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        params.append(p)

    return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_task_iters(
        cfg: Any,
        model: Adapter,
        train_loader: DataLoader,
        device: str,
        use_contrastive: bool = True,
        w: float = 0.5,
        ewc_mean: Optional[dict[str, Tensor]] = None, 
        ewc_fisher: Optional[dict[str, Tensor]] = None,
        ewc_lambda: float = 0.0,   
    ) -> float:

    model.train()
    opt = build_optimizer(cfg, model)

    total_loss = 0.0
    total_steps = 0

    loader_iter = iter(train_loader)

    while total_steps < cfg.iters_per_task:

        try:
            images, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            images, labels = next(loader_iter)

        images = images.to(device)
        labels = labels.to(device)

        features, logits = model(images)

        classif_loss = F.cross_entropy(logits, labels)

        if use_contrastive:
            contr_loss = contrastive_loss(features, labels)
            loss = classif_loss + w * contr_loss
        else:
            loss = classif_loss

        if ewc_mean is not None and ewc_fisher is not None and ewc_lambda > 0.0:
            ewc_loss = ewc_penalty(model, ewc_mean, ewc_fisher)
            loss = loss + ewc_lambda * ewc_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_steps += 1

        if (total_steps) % 200 == 0:
            logger.info(f"    step {total_steps}/{cfg.iters_per_task} | loss {total_loss/max(1, total_steps):.4f}")

    return total_loss / max(1, cfg.iters_per_task)
