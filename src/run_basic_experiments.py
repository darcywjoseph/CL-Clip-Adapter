from dataclasses import dataclass
from typing import Union, Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import numpy as np
import argparse
from model import Adapter
from dataset import ShapesAndColours
from train import train_single_epoch, compute_fisher_diagonal
import logging
import sys
from pathlib import Path
from datetime import datetime

@dataclass
class ExperimentConfig:
    training_epochs: int
    learning_rate: float
    batch_size: int
    num_samples_train: int
    num_samples_test: int
    device: str
    use_contrastive: bool = True
    use_ewc: bool = False
    ewc_lambda: float = 10.0
    fisher_sample_size: int = 200

def evaluate_model(
        model: Adapter, 
        data_loader: DataLoader, 
        device: Union[str, torch.device]
    ) -> dict[str, object]:

    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_features = []

    with torch.no_grad():

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            features, logits = model(images)
            predictions = logits.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(features.cpu().numpy())

        accuracy = total_correct / total_samples
        
        return {
            'accuracy': accuracy,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'features': np.concatenate(all_features, axis=0)
        }
           

def run_basic_experiment(cfg: ExperimentConfig):
    """
    Runs basic experiment on the ShapesAndColours dataset
    using a frozen CLIP backbone with a trainable adapter.

    Experiment tests whether the model can adapt to a task shift
    where the discriminative feature changes across tasks. 
    This is achieved via analysis across 3 tasks.

    Task 1:
        The model is trained to classify objects using either shape and colour.

    Task 2:
        The model is trained to classify when shape becomes non-discriminative
        and colour is the only discriminative feature.

    Task 3:
        Used to evaluate whether the model
        has retained the ability to discrminate based on shape after Task 2.
    """
        
    clip_model, preprocessing = clip.load(
        'ViT-B/16', 
        device=cfg.device,
    )

    clip_plus_adapter = Adapter(clip_model, input_dim=512, hidden_dim=256, output_dim=512)
    clip_plus_adapter = clip_plus_adapter.to(cfg.device)

    dataset_task_1_train = ShapesAndColours(task_id=1, transform=preprocessing, num_samples=cfg.num_samples_train)
    data_loader_task1_train = DataLoader(dataset_task_1_train, batch_size=cfg.batch_size, shuffle=True)
    dataset_task_1_test = ShapesAndColours(task_id=1, transform=preprocessing, num_samples=cfg.num_samples_test)
    data_loader_task1_test = DataLoader(dataset_task_1_test, batch_size=cfg.batch_size, shuffle=False)

    dataset_task_2_train = ShapesAndColours(task_id=2, transform=preprocessing, num_samples=cfg.num_samples_train)
    data_loader_task2_train = DataLoader(dataset_task_2_train, batch_size=cfg.batch_size, shuffle=True)
    dataset_task_2_test = ShapesAndColours(task_id=2, transform=preprocessing, num_samples=cfg.num_samples_test)
    data_loader_task2_test = DataLoader(dataset_task_2_test, batch_size=cfg.batch_size, shuffle=False)


    optimizer = optim.Adam(clip_plus_adapter.parameters(), lr=cfg.learning_rate)

    logger.info("Trainig Task 1...")
    for epoch in range(cfg.training_epochs):
        train_loss = train_single_epoch(
            data_loader_task1_train, 
            clip_plus_adapter, optimizer, 
            use_contrastive=cfg.use_contrastive,
            device=cfg.device
            )
        logger.info(f"epoch {epoch+1}/{cfg.training_epochs} loss: {train_loss:.4f}")


    t1_after_t1 = evaluate_model(clip_plus_adapter, data_loader_task1_test, device=cfg.device)["accuracy"]
    logger.info(f"Task1 accuracy after Task1: {t1_after_t1:.4f}")

    t2_before_t2 = evaluate_model(clip_plus_adapter, data_loader_task2_test, device=cfg.device)["accuracy"]
    logger.info(f"Task2 accuracy before Task2 training: {t2_before_t2:.4f}")

    ewc_mean = None
    ewc_fisher = None
    if cfg.use_ewc:
        logger.info("Computing Fisher (EWC) after Task 1...")
        ewc_mean, ewc_fisher = compute_fisher_diagonal(
            model=clip_plus_adapter,
            dataloader=data_loader_task1_train,
            device=cfg.device,
            fisher_sample_size=cfg.fisher_sample_size,
        )

    logger.info("Training Task 2...")
    for epoch in range(cfg.training_epochs):
        train_loss = train_single_epoch(
            data_loader_task2_train, 
            clip_plus_adapter, 
            optimizer, 
            use_contrastive=cfg.use_contrastive,
            device=cfg.device,
            ewc_mean=ewc_mean,
            ewc_fisher=ewc_fisher,
            ewc_lambda=cfg.ewc_lambda
        )
        logger.info(f"epoch {epoch+1}/{cfg.training_epochs} loss: {train_loss:.4f}")

    # Test model on task 2
    t2_after_t2 = evaluate_model(clip_plus_adapter, data_loader_task2_test, device=cfg.device)["accuracy"]
    logger.info(f"Task2 accuracy after Task2: {t2_after_t2:.4f}")

    # Check forgetting by testing on task 1
    t1_after_t2 = evaluate_model(clip_plus_adapter, data_loader_task1_test,device=cfg.device)["accuracy"]
    accuracy_diff = t1_after_t1 - t1_after_t2
    logger.info(f"Task1 accuracy after Task2: {t1_after_t2:.4f}")
    logger.info(f"Difference in task 1 accuracy from before and after task 2 training: {accuracy_diff:.4f}")

    # Check if model can classify based upon shape by testing on task 3
    dataset_task_3_test = ShapesAndColours(task_id=3, transform=preprocessing, num_samples=cfg.num_samples_test)
    data_loader_task3_test = DataLoader(dataset_task_3_test, batch_size=cfg.batch_size, shuffle=False)

    t3_after_t2 = evaluate_model(clip_plus_adapter, data_loader_task3_test, device=cfg.device)["accuracy"]
    logger.info(f"Task3 accuracy after Task2: {t3_after_t2:.4f}")


if __name__ == "__main__":

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"basic_experiment_{timestamp}.txt"

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
    parser.add_argument("--use_ewc", action="store_true")

    args = parser.parse_args()

    cfg = ExperimentConfig(
        training_epochs=3,
        learning_rate=1e-3,
        batch_size=32,
        num_samples_train=500,
        num_samples_test=50,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_contrastive=args.use_contrastive,
        use_ewc=args.use_ewc,
        ewc_lambda = 10.0,
        fisher_sample_size = 200,
    )

    logger.info("==== Basic Experiment Start ====")
    logger.info(f"Config: {cfg}")

    run_basic_experiment(cfg)

    logger.info("==== Basic Experiment Complete ====")