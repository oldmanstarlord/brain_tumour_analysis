from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

from utils.gradcam import generate_gradcam_grid
from utils.transforms import get_val_transforms


CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


def _load_image_tensor(image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor


def load_classifier(checkpoint_path, device, num_classes=4):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    return model.to(device).eval()


def load_segmentation_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def predict_class(model, image_tensor):
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    prediction = int(probabilities.argmax().item())
    confidence = float(probabilities[prediction].item())
    return prediction, confidence, probabilities.cpu().numpy()


@torch.no_grad()
def predict_segmentation_mask(model, image_tensor, threshold=0.5):
    if model is None:
        return None
    logits = model(image_tensor)
    probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probabilities > threshold).astype(np.float32)
    return probabilities, mask


def run_full_pipeline(image_path, classifier, device, segmentation_model=None):
    original_image, image_tensor = _load_image_tensor(image_path, device)
    predicted_class, confidence, probabilities = predict_class(classifier, image_tensor)
    overlay, heatmap = generate_gradcam_grid(classifier, image_tensor.squeeze(0), device)

    segmentation_probabilities = None
    segmentation_mask = None
    if segmentation_model is not None:
        segmentation_probabilities, segmentation_mask = predict_segmentation_mask(
            segmentation_model, image_tensor
        )

    return {
        "original_image": original_image,
        "predicted_class": predicted_class,
        "prediction_label": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": probabilities,
        "gradcam_overlay": overlay,
        "gradcam_heatmap": heatmap,
        "segmentation_probabilities": segmentation_probabilities,
        "segmentation_mask": segmentation_mask,
    }
