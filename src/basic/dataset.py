import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Optional, Callable
from torch import Tensor
from model.model import Adapter

GREEN = (0, 255, 0)
RED = (255, 0, 0)


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
                    colour = GREEN
                else:
                    colour = RED      
            elif self.task_id == 3:
                colour = GREEN
            else:
                raise ValueError(f"Invalid task id: {self.task_id}")

            if self.task_id == 1 or self.task_id == 3:
                is_square = (label == 0)
            elif self.task_id == 2:
                is_square = (np.random.rand() > 0.5)
            else:
                raise ValueError(f"Invalid task id: {self.task_id}")

            img = _make_image(is_square, colour)

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
    
def _make_image(is_square: bool, colour: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    x = 112
    y = 50

    if is_square:
        draw.rectangle([x - y, x - y, x + y, x + y], fill=colour)
    else:
        draw.ellipse([x - y, x - y, x + y, x + y], fill=colour)

    return img

def cos(a, b):
    return (a * b).sum(dim=-1)

def compute_shape_color_sensitivity(
    model: Adapter,
    preprocess,
    device: str,
    repeats: int = 64,
) -> dict[str, float]:
    """
    Uses 4 images repeated many times to get a estimate of how much
    the representation changes when:
      - shape changes (same color)
      - color changes (same shape)

    Returns mean cosine distances (1 - cosine similarity).
    """

    model.eval()

    green_square = _make_image(is_square=True, colour=GREEN)
    green_circle = _make_image(is_square=False, colour=GREEN)
    red_square = _make_image(is_square=True, colour=RED)
    red_circle = _make_image(is_square=False, colour=RED)

    imgs = [green_square, green_circle, red_square, red_circle] * repeats
    batch = torch.stack([preprocess(im) for im in imgs]).to(device)

    with torch.no_grad():
        feats, _ = model(batch)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

    feats = feats.view(repeats, 4, -1)
    GS_f, GC_f, RS_f, RC_f = feats[:, 0, :], feats[:, 1, :], feats[:, 2, :], feats[:, 3, :]

    shape_sim = torch.cat([cos(GS_f, GC_f), cos(RS_f, RC_f)], dim=0)
    color_sim = torch.cat([cos(GS_f, RS_f), cos(GC_f, RC_f)], dim=0)

    shape_dist = (1.0 - shape_sim).mean().item()
    color_dist = (1.0 - color_sim).mean().item()

    return {
        "shape_cosine_dist": shape_dist,
        "color_cosine_dist": color_dist,
        "shape_minus_color": shape_dist - color_dist,
        "shape_over_color": shape_dist / (color_dist + 1e-8),
    }