from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class TumorSegmentationDataset(Dataset):
    """Paired image-mask dataset for binary tumor segmentation.

    Expected structure:
        images_root/<class_name>/*.png|jpg
        masks_root/<class_name>/*.png|jpg

    If the mask folder is flat, provide the same filenames under masks_root.
    """

    def __init__(self, images_root, masks_root, transform=None, mask_transform=None):
        self.images_root = Path(images_root)
        self.masks_root = Path(masks_root)
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = self._collect_pairs()

    def _collect_pairs(self):
        samples = []
        image_files = [
            p for p in self.images_root.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
        for image_path in sorted(image_files):
            relative = image_path.relative_to(self.images_root)
            mask_path = self.masks_root / relative
            if mask_path.exists():
                samples.append((image_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)

        mask = (mask > 0.5).float()
        return image, mask


def dice_score(predictions, targets, eps=1e-6):
    predictions = (predictions > 0.5).float()
    targets = targets.float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + eps) / (union + eps)).mean().item()


def iou_score(predictions, targets, eps=1e-6):
    predictions = (predictions > 0.5).float()
    targets = targets.float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


@torch.no_grad()
def evaluate_segmentation(model, loader, device, threshold=0.5):
    model.eval()
    scores = []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > threshold).float()
        scores.append(iou_score(predictions, masks))
    return float(np.mean(scores)) if scores else 0.0


def plot_segmentation_triplet(image, mask, prediction, title_prefix="Sample"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"{title_prefix} | Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    return fig
