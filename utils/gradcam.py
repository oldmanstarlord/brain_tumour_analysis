import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def tensor_to_numpy(img_tensor):
    # img_tensor: C,H,W normalized
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def find_last_conv_layer(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def generate_gradcam_overlay(model, target_layer, input_tensor, device, eigen_smooth=True):
    model.eval()
    # Newer versions of pytorch-grad-cam do not accept `use_cuda` keyword.
    # Instantiate with model and target_layers only and rely on the input tensor/device.
    cam = GradCAM(model=model, target_layers=[target_layer])
    input_tensor = input_tensor.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None, eigen_smooth=eigen_smooth)
    grayscale_cam = grayscale_cam[0]
    img = tensor_to_numpy(input_tensor[0])
    # un-normalize assumed to be ImageNet; clip
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    heatmap = (grayscale_cam * 255).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = Image.fromarray(visualization)
    return overlay, Image.fromarray(heatmap)


def generate_gradcam_grid(model, input_tensor, device):
    target_layer = find_last_conv_layer(model)
    if target_layer is None:
        raise ValueError("Could not find a convolution layer for Grad-CAM.")
    overlay, heatmap = generate_gradcam_overlay(model, target_layer, input_tensor, device)
    return overlay, heatmap
