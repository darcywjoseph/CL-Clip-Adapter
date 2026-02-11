from dataclasses import dataclass
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import open_clip
import numpy as np
from model import Adapter
from dataset import ShapesAndColours
from train import train_single_epoch

CLIP_CACHE = "../clip_cache"

@dataclass
class ExperimentConfig:
    training_epochs: int
    learning_rate: float
    batch_size: int
    num_samples_train: int
    num_samples_test: int
    device: str

def evaluate_model(model, data_loader, device):
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
           

def run_basic_experiment(cfg):
    
    clip_model, _, preprocessing = open_clip.create_model_and_transforms(
        'ViT-B-16', 
        pretrained='openai', 
        device=cfg.device,
        cache_dir=CLIP_CACHE
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

    print("Trainig Task 1...")
    for epoch in range(cfg.training_epochs):
        train_loss = train_single_epoch(data_loader_task1_train, clip_plus_adapter, optimizer)
        print(f"epoch {epoch+1}/{cfg.training_epochs} loss: {train_loss:.4f}")


    t1_after_t1 = evaluate_model(clip_plus_adapter, data_loader_task1_test, device=cfg.device)["accuracy"]
    print(f"Task1 accuracy after Task1: {t1_after_t1:.4f}")

    t2_before_t2 = evaluate_model(clip_plus_adapter, data_loader_task2_test, device=cfg.device)["accuracy"]
    print(f"Task2 accuracy before Task2 training: {t2_before_t2:.4f}")

    print("Training Task 2...")
    for epoch in range(cfg.training_epochs):
        train_loss = train_single_epoch(data_loader_task2_train, clip_plus_adapter, optimizer)
        print(f"epoch {epoch+1}/{cfg.training_epochs} loss: {train_loss:.4f}")

    # Test model on task 2
    t2_after_t2 = evaluate_model(clip_plus_adapter, data_loader_task2_test, device=cfg.device)["accuracy"]
    print(f"Task2 accuracy after Task2: {t2_after_t2:.4f}")

    # Check forgetting by testing on task 1
    t1_after_t2 = evaluate_model(clip_plus_adapter, data_loader_task1_test,device=cfg.device)["accuracy"]
    accuracy_diff = t1_after_t1 - t1_after_t2
    print(f"Task1 accuracy after Task2: {t1_after_t2:.4f}")
    print(f"Difference in task 1 accuracy from before and after task 2 training: {accuracy_diff:.4f}")


if __name__ == "__main__":

    cfg = ExperimentConfig(
        training_epochs=3,
        learning_rate=1e-3,
        batch_size=32,
        num_samples_train=500,
        num_samples_test=50,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    run_experiment(cfg)