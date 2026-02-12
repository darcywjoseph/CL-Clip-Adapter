from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
import clip
from pathlib import Path
from mtil.dataset_utils import setup_task_datasets, infer_num_classes
from model.model import Adapter 
from model.train import train_task_iters
import logging
import sys
from datetime import datetime
import argparse

CLIP_CACHE = Path("../clip_cache")

@dataclass
class MTILExperimentConfig:
    data_root: Path = Path("../data")
    clip_model: str = "ViT-B/16"
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    iters_per_task: int = 1000 
    eval_batch_size: int = 128
    seed: int = 0
    order: str = "order_i"
    train_adapter: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_contrastive: bool = False


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_task(
    model: Adapter,
    test_loader: DataLoader,
    device: str,
) -> float:
    
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        _, logits = model(images)

        pred = logits.argmax(dim=1)

        correct += (pred == labels).sum().item()
        total += labels.numel()
        
    return correct / max(1, total)


@torch.no_grad()
def evaluate_zeroshot_clip(
    clip_model,
    test_dataset,
    classnames: list[str],
    device: str,
    batch_size: int = 128,
) -> float:
    """
    Get Baseline zeroshot CLIP accuracy across all tasks.
    """
    clip_model.eval()

    prompts = [f"a photo of a {c}" for c in classnames]
    text = clip.tokenize(prompts).to(device)
    text_feats = clip_model.encode_text(text).float()
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        img_feats = clip_model.encode_image(images).float()
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logits = img_feats @ text_feats.t()
        pred = logits.argmax(dim=1)

        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / max(1, total)

def main(cfg, logger):

    set_seeds(cfg.seed)

    CLIP_CACHE.mkdir(parents= True, exist_ok=True)
    cfg.data_root.mkdir(parents= True, exist_ok=True)

    tasks_order_i = [
        "Aircraft", "CIFAR100", "DTD", "EuroSAT",
        "Flowers", "Food", "MNIST", "OxfordPet", 
    ]

    if cfg.order.lower() == "order_ii":
        tasks = list(reversed(tasks_order_i))
    else:
        tasks = tasks_order_i

    logger.info("Loading CLIP...")
    clip_model, preprocess = clip.load(
        cfg.clip_model,
        device=cfg.device,
    )
    clip_model.eval()

    model = Adapter(clip_model, input_dim=512, hidden_dim=256, output_dim=512).to(cfg.device)

    task_heads: dict[str, dict[str, torch.Tensor]] = {}
    task_num_classes: dict[str, int] = {}
    task_test_loaders: dict[str, DataLoader] = {}

    # Transfer (zero-shot CLIP) across all tasks before training
    logger.info("zero-shot CLIP on all tasks:")
    transfer_accs = []
    for task in tasks_order_i:
        train_ds, test_ds, classnames = setup_task_datasets(task, cfg.data_root, preprocess)
        acc_zs = evaluate_zeroshot_clip(
            clip_model=clip_model,
            test_dataset=test_ds,
            classnames=classnames,
            device=cfg.device,
            batch_size=cfg.eval_batch_size,
        )
        logger.info(f"    {task:10s}: {acc_zs*100:.2f}%")
        transfer_accs.append(acc_zs)
    transfer = sum(transfer_accs) / len(transfer_accs)
    logger.info(f"Transfer (mean zero-shot): {transfer*100:.2f}%")

    acc_matrix: list[list[Optional[float]]] = [] 

    logger.info("=== Continual Training (MTIL) ===")
    for t_idx, task in enumerate(tasks):
        logger.info(f"[Task {t_idx+1}/{len(tasks)}] {task}")

        train_ds, test_ds, classnames = setup_task_datasets(task, cfg.data_root, preprocess)
        n_classes = infer_num_classes(train_ds)
        task_num_classes[task] = n_classes

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        task_test_loaders[task] = test_loader

        # Swap in a new classifier head for task
        model.classifier = nn.Linear(512, n_classes).to(cfg.device)
        
        for name, p in model.named_parameters():
            if "clip_model" in name:
                p.requires_grad = False

        avg_loss = train_task_iters(cfg, model, train_loader, cfg.device, use_contrastive=cfg.use_contrastive)
        logger.info(f"  Finished {task} | avg loss {avg_loss:.4f}")

        # Snapshot this head
        task_heads[task] = {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()}

        row = []
        logger.info("  Eval on seen tasks:")
        seen_tasks = tasks[: t_idx + 1]
        for eval_task in seen_tasks:
            # load the correct head
            head_sd = task_heads[eval_task]
            model.classifier = nn.Linear(512, task_num_classes[eval_task]).to(cfg.device)
            model.classifier.load_state_dict({k: v.to(cfg.device) for k, v in head_sd.items()})

            acc = evaluate_task(model, task_test_loaders[eval_task], cfg.device)
            logger.info(f"    {eval_task:10s}: {acc*100:.2f}%")
            row.append(acc)

        acc_matrix.append(row)

    logger.info("=== Final Evaluation ===")
    last_accs = []
    for task in tasks:
        model.classifier = nn.Linear(512, task_num_classes[task]).to(cfg.device)
        model.classifier.load_state_dict({k: v.to(cfg.device) for k, v in task_heads[task].items()})
        acc = evaluate_task(model, task_test_loaders[task], cfg.device)
        last_accs.append(acc)

    last = sum(last_accs) / len(last_accs)
    avg_metric = 0.5 * (transfer + last)

    logger.info(f"Last (mean final acc): {last*100:.2f}%")
    logger.info(f"Average (Transfer+Last)/2: {avg_metric*100:.2f}%")

    logger.info("\nAccuracy matrix (rows = after learning task i, cols = tasks seen so far):")
    for i, row in enumerate(acc_matrix):
        row_str = " ".join([f"{a*100:6.2f}" for a in row])
        logger.info(f"  after {tasks[i]:10s}: {row_str}")

if __name__ == "__main__":

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"mtil_experiment_{timestamp}.txt"

    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_contrastive", action="store_true")

    args = parser.parse_args()

    cfg = MTILExperimentConfig(
            clip_model = "ViT-B/16",
            lr = 1e-3,
            order = "order_i",
            train_adapter = True,
            use_contrastive = args.use_contrastive,
    )
    
    logger.info("==== MTIL Experiment Start ====")
    logger.info(f"Config: {cfg}")

    main(cfg, logger)

    logger.info("==== MTIL Experiment Complete ====")
