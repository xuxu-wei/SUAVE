from torch import nn
import torch

class ResetMixin:
    def reset_parameters(self, seed=20201021):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.manual_seed(seed)
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)