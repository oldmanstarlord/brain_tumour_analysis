from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import torch

from utils.transforms import get_train_transforms, get_val_transforms


def build_transforms():
    return {
        "train": get_train_transforms(),
        "val": get_val_transforms(),
        "test": get_val_transforms(),
    }


def load_sample_image(root_dir, class_name=None):
    root_dir = Path(root_dir)
    class_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {root_dir}")

    chosen_dir = None
    if class_name is not None:
        for d in class_dirs:
            if d.name.lower() == class_name.lower():
                chosen_dir = d
                break
    if chosen_dir is None:
        chosen_dir = class_dirs[0]

    candidates = [p for p in chosen_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not candidates:
        raise FileNotFoundError(f"No images found in {chosen_dir}")
    return candidates[0], chosen_dir.name


def visualize_transform(image_path, transform, title=None):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    if tensor.ndim == 3:
        display = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        display = tensor.cpu().numpy()

    display = display.copy()
    if display.ndim == 3:
        display = (display - display.min()) / (display.max() - display.min() + 1e-8)

    plt.figure(figsize=(4, 4))
    plt.imshow(display)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()


def get_dataloader_kwargs():
    return {
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
