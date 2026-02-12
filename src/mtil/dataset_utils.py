import torch
from torchvision import datasets
from torch.utils.data import random_split

def split_helper(full_dataset, train_size: int, test_size: int, seed: int=36):

    n = len(full_dataset)

    g = torch.Generator().manual_seed(seed)
    
    train_subset, test_subset, _ = random_split(
        full_dataset, [train_size, test_size, n - train_size - test_size], generator=g
    )
    return train_subset, test_subset
    
def setup_task_datasets(
    task_name: str,
    root: str,
    preprocess,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, list[str]]:

    if task_name == "Aircraft":
        train_split = datasets.FGVCAircraft(root=root, split="train", download=True, transform=preprocess)
        test_split  = datasets.FGVCAircraft(root=root, split="test",  download=True, transform=preprocess)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(len(set(train_split.targets)))]

    elif task_name == "Caltech101":
        # MTIL fixed sizes: 6941 / 1736
        full = datasets.Caltech101(root=root, download=True, transform=preprocess)
        train_split, test_split = split_helper(full, train_size=6941, test_size=1736)
        classnames = getattr(full, "categories", None)

    elif task_name == "CIFAR100":
        train_split = datasets.CIFAR100(root=root, train=True,  download=True, transform=preprocess)
        test_split  = datasets.CIFAR100(root=root, train=False, download=True, transform=preprocess)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(100)]

    elif task_name == "DTD":
        train_split = datasets.DTD(root=root, split="train", download=True, transform=preprocess)
        test_split  = datasets.DTD(root=root, split="test",  download=True, transform=preprocess)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(len(set(train_split.labels)))]

    elif task_name == "EuroSAT":
        full = datasets.EuroSAT(root=root, download=True, transform=preprocess)
        train_split, test_split = split_helper(full, train_size=21600, test_size=5300,)
        classnames = getattr(full, "classes", None) or [f"class_{i}" for i in range(len(full.class_to_idx))]

    elif task_name == "Flowers":
        train_split = datasets.Flowers102(root=root, split="train", download=True, transform=preprocess)
        test_split  = datasets.Flowers102(root=root, split="test",  download=True, transform=preprocess)
        classnames = getattr(train_split, "classes", None) or [f"flower_{i}" for i in range(102)]

    elif task_name == "Food":
        train_split = datasets.Food101(root=root, split="train", download=True, transform=preprocess)
        test_split  = datasets.Food101(root=root, split="test",  download=True, transform=preprocess)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(101)]

    elif task_name == "MNIST":
        train_split = datasets.MNIST(root=root, train=True,  download=True, transform=preprocess)
        test_split  = datasets.MNIST(root=root, train=False, download=True, transform=preprocess)
        classnames = [str(i) for i in range(10)]

    elif task_name == "OxfordPet":
        train_split = datasets.OxfordIIITPet(root=root, split="trainval", download=True, transform=preprocess)
        test_split  = datasets.OxfordIIITPet(root=root, split="test",     download=True, transform=preprocess)
        classnames = getattr(train_split, "classes", None) or [f"pet_{i}" for i in range(len(train_split.class_to_idx))]

    elif task_name == "StanfordCars":
        train_split = datasets.StanfordCars(root=root, split="train", download=True, transform=preprocess)
        test_split  = datasets.StanfordCars(root=root, split="test",  download=True, transform=preprocess)
        classnames = getattr(train_split, "classes", None)

    elif task_name == "SUN397":
        # MTIL fixed sizes: 87003 / 21751
        full = datasets.SUN397(root=root, download=True, transform=preprocess)
        train_split, test_split = split_helper(full, train_size=87003, test_size=21751)
        classnames = getattr(full, "classes", None)


    else:
        raise ValueError(f"Unknown task: {task_name}")

    n_classes = infer_num_classes(train_split)
    if classnames is None or len(classnames) != n_classes:
        classnames = [f"class_{i}" for i in range(n_classes)]

    return train_split, test_split, classnames

def infer_num_classes(dataset: torch.utils.data.Dataset) -> int:
   
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return len(dataset.classes)
    if hasattr(dataset, "class_to_idx") and dataset.class_to_idx is not None:
        return len(dataset.class_to_idx)
    if hasattr(dataset, "categories") and dataset.categories is not None:
        return len(dataset.categories)
    if hasattr(dataset, "targets"):
        try:
            return int(max(dataset.targets)) + 1
        except Exception:
            pass
    if hasattr(dataset, "labels"):
        try:
            return int(max(dataset.labels)) + 1
        except Exception:
            pass
    if hasattr(dataset, "y"):
        try:
            return int(max(dataset.y)) + 1
        except Exception:
            pass

    return 2
