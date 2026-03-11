import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from video at specified FPS
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to FaceForensics++ directory')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract')
    parser.add_argument('--compression', default='c23', choices=['c0', 'c23', 'c40'])
    args = parser.parse_args()
    
    # Paths to process
    video_dirs = [
        f"{args.data_path}/original_sequences/youtube/{args.compression}/videos",
        f"{args.data_path}/manipulated_sequences/Deepfakes/{args.compression}/videos",
        f"{args.data_path}/manipulated_sequences/Face2Face/{args.compression}/videos",
        f"{args.data_path}/manipulated_sequences/FaceSwap/{args.compression}/videos",
        f"{args.data_path}/manipulated_sequences/NeuralTextures/{args.compression}/videos",
    ]
    
    for video_dir in video_dirs:
        if not os.path.exists(video_dir):
            print(f"⚠️  Skipping {video_dir} (not found)")
            continue
        
        output_dir = video_dir.replace('videos', 'images')
        print(f"\n📁 Processing {video_dir}")
        
        video_files = list(Path(video_dir).glob("*.mp4"))
        for video_file in tqdm(video_files, desc='Extracting frames'):
            video_id = video_file.stem
            frame_output_dir = os.path.join(output_dir, video_id)
            
            num_frames = extract_frames(str(video_file), frame_output_dir, fps=args.fps)
        
        print(f"✅ Extracted frames to {output_dir}")

if __name__ == '__main__':
    main()