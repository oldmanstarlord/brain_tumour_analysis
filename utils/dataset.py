import os
from PIL import Image
from torch.utils.data import Dataset


class BrainTumorImageDataset(Dataset):
    """Simple dataset loader expecting directory structure:
    root/class_x/xxx.png
    root/class_y/yyy.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
