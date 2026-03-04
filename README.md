# Deepfake Project - COMP6341

## Dataset Setup

The dataset (FaceForensics++) is not stored in the repository due to its size. Each collaborator must download it locally.

### Download

```bash
pip install tqdm
python3 data/download_script.py data -d all -c c23 -t videos --server EU2
```

This will populate the following directories:

```
data/
├── download_script.py
├── original_sequences/       # Real videos
│   └── actors/c23/videos/
└── manipulated_sequences/    # Deepfake videos
    └── DeepFakeDetection/c23/videos/
```

### Usage in Code

Reference the dataset using relative paths from the project root:

```python
import os

DATA_DIR = os.path.join("data")
REAL_VIDEOS = os.path.join(DATA_DIR, "original_sequences", "actors", "c23", "videos")
FAKE_VIDEOS = os.path.join(DATA_DIR, "manipulated_sequences", "DeepFakeDetection", "c23", "videos")
```

### Download  ONLY c23 compression level
```
python download-FaceForensics.py \
    /path/to/output/directory \
    -d FaceForensics++ \
    -c c23 \
    -t videos
```

This downloads ~38GB instead of ~500GB

### EfficientNet-B1 Usage using timm from Pytorch
```
# Install
pip install timm torch torchvision

# Load pretrained EfficientNet-B1
import timm
import torch.nn as nn

# Create model with ImageNet pretrained weights
model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=2)

# Or for more control:
model = timm.create_model(
    'efficientnet_b1', 
    pretrained=True, 
    num_classes=2,
    drop_rate=0.2,        # Dropout rate
    drop_path_rate=0.2    # Stochastic depth
)

# Model info
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
# Output: Parameters: 7.79M
```
