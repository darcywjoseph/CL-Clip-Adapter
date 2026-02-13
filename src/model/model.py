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
        residual_weight: float = 0.2
    ) -> None:
        
        super(Adapter, self).__init__()
        self.clip_model = clip_model
        self.residual_weight = residual_weight
        
        # Freeze CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Bottleneck Adapter seems to be pretty standard
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.classifier: nn.Module | None = None

    def set_classifier(self, classifier: nn.Module) -> None:
        self.classifier = classifier
        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:

        with torch.no_grad():
            original_features = self.clip_model.encode_image(x).float()

        adapter_out = self.adapter(original_features)

        features = self.residual_weight * adapter_out + (1 - self.residual_weight) * original_features
        logits = self.classifier(features)
        
        return features, logits
    
class CLIPZeroShotHead(nn.Linear):
    """
    adapted from MoE head builidng logic. 
    """
    def __init__(self, weights: torch.Tensor, normalize: bool = True):
        # weights: (num_classes, feat_dim)
        out_dim, in_dim = weights.shape
        super().__init__(in_dim, out_dim, bias=False)
        self.normalize = normalize
        with torch.no_grad():
            self.weight.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        return super().forward(x)