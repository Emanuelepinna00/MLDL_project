
import torch
import timm
import logging
import torchvision
from torch import nn
from typing import Tuple
import torch

from cosplace_model.layers import Flatten, L2Norm, GeM, AdaptivePooling, AttentionPooling, MixVPR

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "EfficientNet_b0": 1280,
    "EfficientNet_b1": 1280,
    "EfficientNet_b2": 1408,
    "MobileViTV2_050": 256,
    "MobileViTV2_100": 512,
    "MobileViTV2_150": 768,
}

IMAGE_SIZE_AFTER_SECOND_TO_LAST_CONV = {
    "ResNet18": 14,
}
AGGREGATION_LAYERS = {
    "gem",
    "adaptive",
    "attention",
    "mixVPR"
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, aggregation : str, fc_output_dim : int):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone)
        assert aggregation in AGGREGATION_LAYERS
        if args.aggregation == "mixVPR":
            self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])
        self.aggregation = get_aggregation_layer(aggregation)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str) -> Tuple[torch.nn.Module, int]:
    if backbone_name.startswith("ResNet"):
        backbone = get_pretrained_torchvision_model(backbone_name)
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        backbone = get_pretrained_torchvision_model(backbone_name)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug(f"Train last layers of the {backbone_name}, freeze the previous ones")
    
    elif backbone_name.startswith("EfficientNet"):
        backbone = get_pretrained_torchvision_model(backbone_name)
        layers = list(backbone.features.children())  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
          for p in layer.parameters():
              p.requires_grad = False
        logging.debug(f"Train last layers of the {backbone_name}, freeze the previous ones")

    elif backbone_name.startswith("MobileViT"):
        if backbone_name.endswith("150"):
            backbon_name = backbone_name + ".cvnets_in22k_ft_in1k"
        backbone = timm.create_model(backbone_name.lower(), pretrained = True, num_classes=0, global_pool='')
        layers = list(backbone.children())[:-1]  # Remove avg pooling and FC layer
        logging.debug(f"Train last layers of the {backbone_name}, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim
def get_aggregation(aggregation_layer : str):
    previous = [L2Norm()]
    following = [Flatten(),
                 nn.Linear(features_dim, fc_output_dim),
                 L2Norm()]
    if aggregation_layer == "gem":
        aggregation = previous.extend([GeM()]).extend(following)
    elif aggregation_layer == "adaptive":
        aggregation = previous.extend([AdaptivePooling()]).extend(following)
    elif aggregation_layer == "attention":
        aggregation = previous.extend([AttentionPooling(CHANNELS_NUM_IN_LAST_CONV[backbone])]).extend(following)
    elif aggregation_layer == "mixVPR":
        assert args.mixVPR_depth > 0
        aggregation = [L2Norm(),
                       MixVPR(in_channels=int(CHANNELS_NUM_IN_LAST_CONV[backbone]/2),
                               in_h=IMAGE_SIZE_AFTER_SECOND_TO_LAST_CONV[backbone],
                               in_w=IMAGE_SIZE_AFTER_SECOND_TO_LAST_CONV[backbone],
                               out_channels=32,
                               mix_depth=args.VPR_depth,
                               mlp_ratio=1,
                               out_rows=4),
                        Flatten(),
                        L2Norm()]
    return torch.nn.Sequential(aggregation)
    
