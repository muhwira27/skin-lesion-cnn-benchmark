
import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMExplainer:
    def __init__(self, model, target_layers=None, use_cuda=True):
        """
        Args:
            model: PyTorch model (already loaded with weights)
            target_layers: List of layers to visualize. If None, tries to auto-detect.
            use_cuda: dict or bool
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if target_layers is None:
            self.target_layers = self._auto_detect_layers(model)
        else:
            self.target_layers = target_layers
            
        # Initialize GradCAM
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def _auto_detect_layers(self, model):
        """
        Simple heuristic to find the last convolutional layer.
        For rigorous usage, specify layers manually for each backbone.
        """
        # 1. TIMM / Torchvision ResNet
        if hasattr(model, 'layer4'):
            return [model.layer4[-1]]
        
        # 2. EfficientNet (timm)
        if hasattr(model, 'blocks'):
            # Usually the last block
            return [model.blocks[-1]]
            
        # 3. ConvNeXt (timm)
        if hasattr(model, 'stages'):
            return [model.stages[-1]]
            
        # 4. MobileNetV3 (torchvision)
        if hasattr(model, 'features'):
            return [model.features[-1]]
            
        # Fallback: try to find the last module that is a Conv2d
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                layers.append(module)
        if layers:
            return [layers[-1]]
            
        raise ValueError("Could not auto-detect target layers. Please specify `target_layers` manually.")

    def explain(self, input_tensor, original_image, target_category_int=None):
        """
        Args:
            input_tensor: Tensor of shape (1, C, H, W) - normalized
            original_image: Numpy array (H, W, 3) - float32 [0, 1] RGB
            target_category_int: Integer class index. If None, uses top prediction.
            
        Returns:
            heatmap: The raw heatmap (grayscale)
            viz_image: Heatmap overlaid on original image
        """
        targets = None
        if target_category_int is not None:
            targets = [ClassifierOutputTarget(target_category_int)]

        # Generate heatmap
        # grayscale_cam shape: (batch_size, H, W)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Take the first item in batch
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay
        # show_cam_on_image expects image in [0, 1]
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
        return grayscale_cam, visualization

def get_target_layer_name(model_name):
    """
    Helper to return a string description of the layer, 
    useful if we want to print what we are visualizing.
    """
    if "resnet" in model_name:
        return "layer4[-1]"
    elif "efficientnet" in model_name:
        return "blocks[-1]"
    elif "convnext" in model_name:
        return "stages[-1]"
    return "last_conv"
