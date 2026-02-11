import torch 
from torch import Tensor
import torch.nn as nn

class Adapter(nn.Module):
    """
    A trainable adapter on top of a frozen CLIP encoder.
    """
    def __init__(
        self,
        clip_model: nn.Module,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
    ) -> None:
        
        super(Adapter, self).__init__()
        self.clip_model = clip_model
        
        # Freeze CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Bottleneck Adapter seems to be pretty standard
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.layer_norm = nn.LayerNorm(output_dim) # TODO: test without this. 
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:

        with torch.no_grad():
            features = self.clip_model.encode_image(x).float()
        #TODO: Maybe add a residual connection here. 
        features = self.adapter(features)
        logits = self.classifier(features)
        
        return features, logits