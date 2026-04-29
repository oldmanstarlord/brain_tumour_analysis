from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np


def get_class_distribution(root):
    root = Path(root)
    classes = [d for d in root.iterdir() if d.is_dir()]
    counts = {}
    samples = {}
    for c in classes:
        imgs = list(c.glob('*'))
        imgs = [p for p in imgs if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        counts[c.name] = len(imgs)
        samples[c.name] = imgs[:5]
    return counts, samples


def plot_class_distribution(counts, figsize=(8,4)):
    sns.set(style='whitegrid')
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    plt.figure(figsize=figsize)
    sns.barplot(x=labels, y=vals)
    plt.ylabel('Image count')
    plt.title('Class distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()


def show_samples_grid(samples, per_class=3, img_size=(224,224)):
    # samples: dict[class_name] -> list[Path]
    classes = list(samples.keys())
    n_classes = len(classes)
    per_class = min(per_class, max(len(v) for v in samples.values()))
    fig, axs = plt.subplots(n_classes, per_class, figsize=(per_class*3, n_classes*3))
    if n_classes == 1:
        axs = [axs]
    for i, cls in enumerate(classes):
        imgs = samples[cls][:per_class]
        for j in range(per_class):
            ax = axs[i][j] if n_classes > 1 else axs[j]
            if j < len(imgs):
                im = Image.open(imgs[j]).convert('RGB')
                im = im.resize(img_size)
                ax.imshow(np.array(im))
                ax.axis('off')
            else:
                ax.axis('off')
            if j == 0:
                ax.set_title(cls, loc='left')
    plt.tight_layout()
