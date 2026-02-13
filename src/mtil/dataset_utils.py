import torch
from torchvision import datasets

from mtil.dataset_config import (
    Aircraft,
    Caltech101,
    CIFAR100,
    DTD,
    EuroSAT,
    Flowers,
    Food,
    MNIST,
    OxfordPet,
    StanfordCars,
    SUN397,
)
    
def setup_task_datasets(
    task_name: str,
    root: str,
    preprocess,
    batch_size: int,
    batch_size_eval: int,
    num_workers: int,
    append_dataset_name_to_template: bool = False,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, list[str]]:
        
    registry = {
        "Aircraft": Aircraft,
        "Caltech101": Caltech101,
        "CIFAR100": CIFAR100,
        "DTD": DTD,
        "EuroSAT": EuroSAT,
        "Flowers": Flowers,
        "Food": Food,
        "MNIST": MNIST,
        "OxfordPet": OxfordPet,
        "StanfordCars": StanfordCars,
        "SUN397": SUN397
    }

    if task_name not in registry:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(registry.keys())}")
    
    mtil_datasets = registry[task_name](
        preprocess=preprocess,
        location=root,
        batch_size=batch_size,
        batch_size_eval=batch_size_eval,
        num_workers=num_workers,
        append_dataset_name_to_template=append_dataset_name_to_template
    )

    train_dataset = mtil_datasets.train_dataset
    test_dataset = mtil_datasets.test_dataset
    classnames = list(mtil_datasets.classnames)
    templates = list(mtil_datasets.templates) 

    return train_dataset, test_dataset, classnames, templates
