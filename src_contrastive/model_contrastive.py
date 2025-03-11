import torch
from lightly.models.modules import heads

class SimCLR(torch.nn.Module):
    def __init__(
            self,
            backbone,
            input_dim=2048,  # Default for ResNet50 (512 for ResNet18/34)
            hidden_dim=2048,
            output_dim=128
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z