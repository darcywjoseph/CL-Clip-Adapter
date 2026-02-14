from dataclasses import dataclass
from typing import Sequence, Callable, Union, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
import clip
from pathlib import Path
from mtil.dataset_utils import setup_task_datasets
from model.model import Adapter 
from model.train import train_task_iters, compute_fisher_diagonal
from model.model_utils import get_prompts, build_clip_zeroshot_head, CLIPZeroShotHead
import logging
import sys
from datetime import datetime
import argparse

TemplateType = Union[str, Callable[[str], str]]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
CLIP_CACHE = Path("../clip_cache")

@dataclass
class MTILExperimentConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    clip_model: str = "ViT-B/16"
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    iters_per_task: int = 1000 
    eval_batch_size: int = 128
    seed: int = 0
    order: str = "order_i"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_contrastive: bool = False
    use_ewc: bool = False
    residual_weight: float = 0.2
    skip_zeroshot: bool = False
    ewc_lambda: float = 10.0
    fisher_sample_size: int = 200


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_transfer_upper_right(acc_full_matrix: list[list[float]]) -> float:

    N = len(acc_full_matrix)
    per_task = []
    for j in range(N):
        vals = [acc_full_matrix[i][j] for i in range(j)]
        if len(vals) > 0:
            per_task.append(sum(vals) / len(vals))
    return sum(per_task) / max(1, len(per_task))

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
    templates: Sequence[Union[str, Callable[[str], str]]] | None = None,
) -> float:
    clip_model.eval()

    text_feats = []
    for cname in classnames:
        prompts = get_prompts(cname, templates)
        text = clip.tokenize(prompts).to(device)
        emb = clip_model.encode_text(text).float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        emb = emb.mean(dim=0)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        text_feats.append(emb)

    text_feats = torch.stack(text_feats, dim=0)

    logit_scale = 1.0
    if hasattr(clip_model, "logit_scale"):
        logit_scale = clip_model.logit_scale.exp().float()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        img_feats = clip_model.encode_image(images).float()
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)

        logits = logit_scale * (img_feats @ text_feats.t())
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / max(1, total)


def main(cfg, logger):

    set_seeds(cfg.seed)

    CLIP_CACHE.mkdir(parents= True, exist_ok=True)
    cfg.data_root.mkdir(parents= True, exist_ok=True)

    tasks_order_i = [
        "Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT",
        "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars"]
    tasks_order_ii = ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers",
        "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"
    ]
    if cfg.order == "order_i":
        tasks_in_order = tasks_order_i
    elif cfg.order == "order_ii":
        tasks_in_order = tasks_order_ii
    else:
        raise ValueError("Unknown order config: {cfg.order}")


    logger.info("Loading CLIP...")
    clip_model, preprocess = clip.load(
        cfg.clip_model,
        device=cfg.device, jit=False
    )
    clip_model.eval()

    model = Adapter(clip_model, input_dim=512, hidden_dim=256, output_dim=512, residual_weight=cfg.residual_weight).to(cfg.device)

    task_heads: dict[str, torch.Tensor] = {}
    task_test_loaders: dict[str, DataLoader] = {}
    all_templates: dict[str, list[TemplateType]] = {}
    all_test_loaders: dict[str, DataLoader] = {}

    all_classnames: dict[str, list[str]] = {}

    if not cfg.skip_zeroshot:
        logger.info("zero-shot CLIP on all tasks:")
        all_zs_ac = []
        for task in tasks_in_order:

            train_dataset, test_dataset, classnames, templates = setup_task_datasets(
                task_name=task,
                root=str(cfg.data_root),
                preprocess=preprocess,
                batch_size=cfg.batch_size,
                batch_size_eval=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
            )
            all_classnames[task] = classnames
            all_templates[task] = templates
            all_test_loaders[task] = DataLoader(
                test_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            acc_zs = evaluate_zeroshot_clip(
                clip_model=clip_model,
                test_dataset=test_dataset,
                classnames=classnames,
                device=cfg.device,
                batch_size=cfg.eval_batch_size,
                templates=templates, 
            )

            logger.info(f"    {task:10s}: {acc_zs*100:.2f}%")
            all_zs_ac.append(acc_zs)

        accuracy = sum(all_zs_ac) / len(all_zs_ac)
        logger.info(f"Mean zero-shot accuracy: {accuracy*100:.2f}%")
    else:
        logger.info("Skipping zero-shot CLIP evaluation")

    N = len(tasks_in_order)
    acc_matrix: list[list[float]] = []

    ewc_mean: dict[str, torch.Tensor] | None = None
    ewc_fisher: dict[str, torch.Tensor] | None = None

    logger.info("=== Evaluating CLIP Adapter on MTIL ===")
    for t_idx, task in enumerate(tasks_in_order):
        logger.info(f"[Task {t_idx+1}/{len(tasks_in_order)}] {task}")

        train_dataset, test_dataset, classnames, templates = setup_task_datasets(
            task_name=task,
            root=str(cfg.data_root),
            preprocess=preprocess,
            batch_size=cfg.batch_size,
            batch_size_eval=cfg.eval_batch_size,
            num_workers=cfg.num_workers,
        )

        if task not in all_classnames:
            all_classnames[task] = classnames
            all_templates[task] = templates
            all_test_loaders[task] = DataLoader(
                test_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        task_test_loaders[task] = test_loader

        # Swap in a new classifier head for task
        head = build_clip_zeroshot_head(
            clip_model=clip_model,
            classnames=classnames,
            device=cfg.device,
            templates=templates,
            use_logit_scale=True,
        )
        model.set_classifier(head)
        
        for name, p in model.named_parameters():
            if "clip_model" in name:
                p.requires_grad = False

        avg_loss = train_task_iters(
            cfg, 
            model, 
            train_loader, 
            cfg.device, 
            use_contrastive=cfg.use_contrastive,
            ewc_mean=ewc_mean,
            ewc_fisher=ewc_fisher,
            ewc_lambda=cfg.ewc_lambda)
        
        logger.info(f"  Finished {task} | avg loss {avg_loss:.4f}")

        if cfg.use_ewc:
            logger.info(f"  Computing Fisher information for {task}...")
            task_mean, task_fisher = compute_fisher_diagonal(
                model=model,
                dataloader=train_loader,
                device=cfg.device,
                fisher_sample_size=cfg.fisher_sample_size,
            )
            
            if ewc_mean is None:
                ewc_mean = task_mean
                ewc_fisher = task_fisher
            else:
                # Accumulate Fisher across tasks 
                for name in ewc_fisher:
                    if name in task_fisher:
                        ewc_fisher[name] = ewc_fisher[name] + task_fisher[name]
                # Update mean to current parameters
                ewc_mean = task_mean

        # save this head for later eval
        task_heads[task] = model.classifier.weight.detach().cpu().clone()

        row = [0.0] * N
        logger.info("  Eval on seen and unseen tasks:")
        for j, eval_task in enumerate(tasks_in_order):
            if eval_task in task_heads:
                # load head corresponding to seen task
                W = task_heads[eval_task].to(cfg.device)
            else:
                # fetch zeroshot head for unseen tasks
                # load classnames and templates if not already loaded from zeroshot
                if eval_task not in all_classnames:
                    _, test_ds_tmp, cn_tmp, tpl_tmp = setup_task_datasets(
                        task_name=eval_task,
                        root=str(cfg.data_root),
                        preprocess=preprocess,
                        batch_size=cfg.batch_size,
                        batch_size_eval=cfg.eval_batch_size,
                        num_workers=cfg.num_workers,
                    )
                    all_classnames[eval_task] = cn_tmp
                    all_templates[eval_task] = tpl_tmp
                    all_test_loaders[eval_task] = DataLoader(
                        test_ds_tmp,
                        batch_size=cfg.eval_batch_size,
                        shuffle=False,
                        num_workers=cfg.num_workers,
                        pin_memory=True,
                    )
                
                tmp_head = build_clip_zeroshot_head(
                    clip_model=clip_model,
                    classnames=all_classnames[eval_task],
                    device=cfg.device,
                    templates=all_templates[eval_task],
                    use_logit_scale=True,
                )
                W = tmp_head.weight.detach() 

            model.set_classifier(CLIPZeroShotHead(W, normalize=True).to(cfg.device))
            for p in model.classifier.parameters():
              p.requires_grad = False

            acc = evaluate_task(model, all_test_loaders[eval_task], cfg.device)

            tag = "seen task" if j <= t_idx else "unseen task"
            logger.info(f"    {tag} {eval_task}: {acc*100:.2f}%")

            row[j] = acc

        acc_matrix.append(row)

    logger.info("=== Final Evaluation ===")
    last_accs = []
    for task in tasks_in_order:
        W = task_heads[task].to(cfg.device)
        model.set_classifier(CLIPZeroShotHead(W, normalize=True).to(cfg.device))
        for p in model.classifier.parameters():
            p.requires_grad = False

        acc = evaluate_task(model, all_test_loaders[task], cfg.device)
        last_accs.append(acc)

    last = sum(last_accs) / len(last_accs)
    transfer = compute_transfer_upper_right(acc_matrix)
    avg_metric = 0.5 * (transfer + last)

    logger.info(f"Transfer (upper-right avg): {transfer*100:.2f}%")
    logger.info(f"Last (mean final acc): {last*100:.2f}%")
    logger.info(f"Average (Transfer+Last)/2: {avg_metric*100:.2f}%")

    logger.info("\nFull accuracy matrix (rows=after learning i, cols=task j):")
    for i, row in enumerate(acc_matrix):
        row_str = " ".join([f"{a*100:6.2f}" for a in row])
        logger.info(f"  after {tasks_in_order[i]:10s}: {row_str}")

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

    parser = argparse.ArgumentParser(description="MTIL Experiment with CLIP Adapter")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--iters_per_task", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order", type=str, default="order_i",
                        choices=["order_i", "order_ii"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--residual_weight", type=float, default=0.2)
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--use_ewc", action="store_true")
    parser.add_argument("--skip_zeroshot", action="store_true")
    parser.add_argument("--ewc_lambda", type=float, default=10.0,)
    parser.add_argument("--fisher_sample_size", type=int, default=200)

    args = parser.parse_args()

    cfg = MTILExperimentConfig(
        data_root=Path(args.data_root),
        clip_model=args.clip_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        iters_per_task=args.iters_per_task,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        order=args.order,
        device=args.device,
        use_contrastive=args.use_contrastive,
        use_ewc=args.use_ewc,
        residual_weight=args.residual_weight,
        skip_zeroshot=args.skip_zeroshot,
        ewc_lambda=args.ewc_lambda,
        fisher_sample_size=args.fisher_sample_size,
    )
    
    logger.info("==== MTIL Experiment Start ====")
    logger.info(f"Config: {cfg}")

    main(cfg, logger)

    logger.info("==== MTIL Experiment Complete ====")