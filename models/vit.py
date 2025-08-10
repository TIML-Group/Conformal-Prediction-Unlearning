# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
from timm.models import vit_small_patch16_224

class ViT(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ViT, self).__init__()
        self.model = vit_small_patch16_224(pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)