from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


class ShapesAndColours(Dataset):
    """This class generates a very simple, 
    synthetic dataset composed of squares and circles, either red or green. 
    According to the task (either task 1 or task 2), the dataset is configured differently

    Task 1 - both colour and shape are discriminative features 
            i.e. class 0 is green squares; class 1 is red circles
    Task 2 - Shape becomes non-discrminiative with colour being the sole discrminant feature. 
            i.e. class 0 is green; class 1 is red
    """

    def __init__(self, task_id, transform = None, num_samples=500):
        
        self.task_id = task_id
        self.transform = transform
        self.data = []

        for _ in range(num_samples):

            label = np.random.randint(0,2)

            if label == 0:
                colour = (0,255,0) #green
            else:
                colour = (255,0,0) # red

            if self.task_id == 1:
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

    def save_dataset(self, base_dir="../datasets"):

        save_dir = Path(base_dir) / "ShapesAndColours" / f"task{self.task_id}"

        save_dir.mkdir(parents=True, exist_ok=True)

        for i, (img, label) in enumerate(self.data):
            file_path = save_dir / f"img_{i}_label_{label}.png"
            img.save(file_path)

        print(f"Saved {len(self.data)} images to {save_dir.resolve()}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
if __name__ == "__main__":

    dataset_task_1 = ShapesAndColours(task_id=1,transform=None,num_samples=500)
    dataset_task_1.save_dataset()

    dataset_task_2 = ShapesAndColours(task_id=2,transform=None,num_samples=500)
    dataset_task_2.save_dataset()