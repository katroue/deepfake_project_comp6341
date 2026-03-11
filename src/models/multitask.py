import torch.nn as nn
import timm

class MultiTaskEfficientNet(nn.Module):
    """
    Strategy 6: Multi-task learning
    Predicts both binary (real/fake) and manipulation type
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Backbone (without classifier)
        self.backbone = timm.create_model(
            'efficientnet_b1',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 240, 240)
            features = self.backbone(dummy_input)
            feat_dim = features.shape[1]
        
        # Task 1: Binary classification head
        self.binary_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 2)
        )
        
        # Task 2: Manipulation type classification head (5 classes)
        self.multiclass_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 5)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        binary_out = self.binary_head(features)
        multiclass_out = self.multiclass_head(features)
        return binary_out, multiclass_out
    