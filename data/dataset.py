import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ dataset for deepfake detection
    
    Args:
        data_root: Path to FaceForensics++ root directory
        split: 'train', 'val', or 'test'
        compression: 'c0', 'c23', or 'c40'
        transform: Image transformations
        manipulations: List of manipulation types to include
    """
    
    def __init__(self, data_root, split='train', compression='c23', 
                 transform=None, manipulations=None):
        self.data_root = data_root
        self.split = split
        self.compression = compression
        self.transform = transform
        
        if manipulations is None:
            manipulations = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        
        # Load split file
        with open(os.path.join(data_root, 'splits', f'{split}.json'), 'r') as f:
            video_ids = json.load(f)
        
        # Build sample list
        self.samples = []
        self._build_samples(video_ids, manipulations)
    
    def _build_samples(self, video_ids, manipulations):
        """Build list of (image_path, label, manipulation_type)"""
        
        # Real images (label=0)
        for vid_id in video_ids:
            frame_dir = os.path.join(
                self.data_root, 'raw', 'original_sequences', 
                'youtube', self.compression, 'images', vid_id
            )
            if os.path.exists(frame_dir):
                for frame in os.listdir(frame_dir):
                    if frame.endswith('.jpg'):
                        self.samples.append((
                            os.path.join(frame_dir, frame),
                            0,  # Real
                            'Real'
                        ))
        
        # Fake images (label=1)
        for manip in manipulations:
            for vid_id in video_ids:
                frame_dir = os.path.join(
                    self.data_root, 'raw', 'manipulated_sequences',
                    manip, self.compression, 'images', vid_id
                )
                if os.path.exists(frame_dir):
                    for frame in os.listdir(frame_dir):
                        if frame.endswith('.jpg'):
                            self.samples.append((
                                os.path.join(frame_dir, frame),
                                1,  # Fake
                                manip
                            ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, manip_type = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, manip_type
    
    def get_manipulation_distribution(self):
        """Get distribution of manipulation types"""
        from collections import Counter
        manip_types = [s[2] for s in self.samples]
        return Counter(manip_types)