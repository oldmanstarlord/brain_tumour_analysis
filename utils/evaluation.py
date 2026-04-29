from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    matrix = confusion_matrix(y_true, y_pred)
    if normalize:
        matrix = matrix.astype(float) / matrix.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()


def get_classification_report_text(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names, digits=4)


def compute_per_class_accuracy(y_true, y_pred, class_names):
    per_class_totals = defaultdict(int)
    per_class_correct = defaultdict(int)

    for actual, predicted in zip(y_true, y_pred):
        per_class_totals[actual] += 1
        per_class_correct[actual] += int(actual == predicted)

    results = {}
    for idx, class_name in enumerate(class_names):
        total = per_class_totals[idx]
        correct = per_class_correct[idx]
        results[class_name] = correct / total if total else 0.0
    return results


def plot_per_class_accuracy(per_class_accuracy):
    plt.figure(figsize=(8, 4))
    names = list(per_class_accuracy.keys())
    values = list(per_class_accuracy.values())
    sns.barplot(x=names, y=values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=30)
    plt.tight_layout()
