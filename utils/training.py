from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return EpochMetrics(
        loss=running_loss / max(total, 1),
        accuracy=correct / max(total, 1),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return EpochMetrics(
        loss=running_loss / max(total, 1),
        accuracy=correct / max(total, 1),
    )


def save_checkpoint(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
