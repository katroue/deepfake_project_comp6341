import torch.nn as nn
import timm

class EfficientNetB1(nn.Module):
    """
    EfficientNet-B1 for binary classification
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        
        # Load pretrained EfficientNet-B1
        self.model = timm.create_model(
            'efficientnet_b1',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=0.2
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())