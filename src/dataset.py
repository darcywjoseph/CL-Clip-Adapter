from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Optional, Callable
import torch
from torch import Tensor
from torchvision import datasets

class ShapesAndColours(Dataset):
    """This class generates a very simple, 
    synthetic dataset composed of squares and circles, either red or green. 
    According to the task (either task 1 or task 2), the dataset is configured differently

    Task 1 - both colour and shape are discriminative features 
            i.e. class 0 is green squares; class 1 is red circles
    Task 2 - Shape becomes non-discrminiative with colour being the sole discrminant feature. 
            i.e. class 0 is green; class 1 is red
    Task 3 - Shape becomes discriminat feature, colour is non discriminate
    """

    def __init__(
        self,
        task_id: int,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        num_samples: int = 500,
    ) -> None:
        
        self.task_id = task_id
        self.transform = transform
        self.data = []

        for _ in range(num_samples):

            label = np.random.randint(0,2)

            if self.task_id == 1 or self.task_id == 2:
                if label == 0:
                    colour = (0,255,0) #green
                else:
                    colour = (255,0,0) # red      
            elif self.task_id == 3:
                colour = (0,255,0)
            else:
                raise ValueError(f"Invalid task id: {self.task_id}")

            if self.task_id == 1 or self.task_id == 3:
                is_square = (label == 0)
            elif self.task_id == 2:
                is_square = (np.random.rand() > 0.5)
            else:
                raise ValueError(f"Invalid task id: {self.task_id}")

            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            x = 112
            y = 50

            if is_square:
                draw.rectangle([x-y, x-y, x+y, x+y], fill=colour)
            else:
                draw.ellipse([x-y, x-y, x+y, x+y], fill=colour)

            self.data.append((img, label))

    def save_dataset(self, base_dir: Path = Path("../datasets")) -> None:

        save_dir = base_dir / "ShapesAndColours" / f"task{self.task_id}"

        save_dir.mkdir(parents=True, exist_ok=True)

        for i, (img, label) in enumerate(self.data):
            file_path = save_dir / f"img_{i}_label_{label}.png"
            img.save(file_path)

        print(f"Saved {len(self.data)} images to {save_dir.resolve()}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
def setup_task_datasets(
    task_name: str,
    root: str,
    preprocess_train,
    preprocess_eval,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, list[str]]:

    if task_name == "Aircraft":
        train_split = datasets.FGVCAircraft(root=root, split="train", download=True, transform=preprocess_train)
        test_split  = datasets.FGVCAircraft(root=root, split="test",  download=True, transform=preprocess_eval)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(len(set(train_split.targets)))]
    elif task_name == "CIFAR100":
        train_split = datasets.CIFAR100(root=root, train=True,  download=True, transform=preprocess_train)
        test_split  = datasets.CIFAR100(root=root, train=False, download=True, transform=preprocess_eval)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(100)]

    elif task_name == "DTD":
        train_split = datasets.DTD(root=root, split="train", download=True, transform=preprocess_train)
        test_split  = datasets.DTD(root=root, split="test",  download=True, transform=preprocess_eval)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(len(set(train_split.labels)))]

    elif task_name == "EuroSAT":
        train_split = datasets.EuroSAT(root=root, download=True, transform=preprocess_train)
        test_split  = datasets.EuroSAT(root=root, download=True, transform=preprocess_eval)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(len(train_split.class_to_idx))]

    elif task_name == "Flowers":
        train_split = datasets.Flowers102(root=root, split="train", download=True, transform=preprocess_train)
        test_split  = datasets.Flowers102(root=root, split="test",  download=True, transform=preprocess_eval)
        classnames = getattr(train_split, "classes", None) or [f"flower_{i}" for i in range(102)]

    elif task_name == "Food":
        train_split = datasets.Food101(root=root, split="train", download=True, transform=preprocess_train)
        test_split  = datasets.Food101(root=root, split="test",  download=True, transform=preprocess_eval)
        classnames = getattr(train_split, "classes", None) or [f"class_{i}" for i in range(101)]

    elif task_name == "MNIST":
        train_split = datasets.MNIST(root=root, train=True,  download=True, transform=preprocess_train)
        test_split  = datasets.MNIST(root=root, train=False, download=True, transform=preprocess_eval)
        classnames = [str(i) for i in range(10)]

    elif task_name == "OxfordPet":
        train_split = datasets.OxfordIIITPet(root=root, split="trainval", download=True, transform=preprocess_train)
        test_split  = datasets.OxfordIIITPet(root=root, split="test",     download=True, transform=preprocess_eval)
        classnames = getattr(train_split, "classes", None) or [f"pet_{i}" for i in range(len(train_split.class_to_idx))]

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
    
if __name__ == "__main__":

    dataset_task_1 = ShapesAndColours(task_id=1,transform=None,num_samples=500)
    dataset_task_1.save_dataset()

    dataset_task_2 = ShapesAndColours(task_id=2,transform=None,num_samples=500)
    dataset_task_2.save_dataset()

    dataset_task_3 = ShapesAndColours(task_id=3,transform=None,num_samples=500)
    dataset_task_3.save_dataset()