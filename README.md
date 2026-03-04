# deepfake_project_comp6341

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
