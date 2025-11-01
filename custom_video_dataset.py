import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

class CustomVideoDataset(Dataset):
    def __init__(
        self,
        base_video_dataset_dir: str,  # Changed parameter name to match convention
        enc_preprocessor,
        sam_preprocessor,
        conversation_generator,
        image_set: str = "train",  # Changed from 'split' to match convention
        num_frames_for_sam: int = 8,
        vision_tower=None,
        image_vision_tower=None,
        image_size: int = 224
    ):
        self.data_root = os.path.join(base_video_dataset_dir, image_set)
        self.enc_preprocessor = enc_preprocessor
        self.sam_preprocessor = sam_preprocessor
        self.conversation_generator = conversation_generator
        self.num_frames_for_sam = num_frames_for_sam
        self.vision_tower = vision_tower
        self.image_vision_tower = image_vision_tower
        self.image_size = image_size
        
        # Get all video directories
        self.video_dirs = sorted([
            d for d in os.listdir(self.data_root) 
            if os.path.isdir(os.path.join(self.data_root, d))
        ])
        
        print(f"Found {len(self.video_dirs)} videos in {self.data_root}")
    
    def __len__(self):
        return len(self.video_dirs)
    
    def load_frames_and_masks(self, video_path: str):
        """Load frames and corresponding masks"""
        viz_dir = os.path.join(video_path, "visualizations")
        
        # Get all frame files
        frame_files = sorted([
            f for f in os.listdir(viz_dir) 
            if f.endswith('.jpg') and f.startswith('frame_')
        ])
        
        # Sample frames uniformly
        total_frames = len(frame_files)
        if total_frames <= self.num_frames_for_sam:
            selected_indices = list(range(total_frames))
        else:
            selected_indices = np.linspace(0, total_frames-1, self.num_frames_for_sam, dtype=int)
        
        frames = []
        masks = []
        frame_paths = []
        
        for idx in selected_indices:
            # Load frame
            frame_file = frame_files[idx]
            frame_path = os.path.join(viz_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
            frame_paths.append(frame_path)
            
            # Load corresponding mask
            mask_file = frame_file.replace('.jpg', '.png')
            mask_path = os.path.join(viz_dir, mask_file)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # Convert to binary mask format
                mask_array = np.array(mask)
                if mask_array.max() > 1:
                    mask_array = (mask_array > 127).astype(np.uint8)
                masks.append(mask_array)
            else:
                # Create empty mask if not exists
                h, w = frame.size[::-1]  # PIL returns (width, height)
                mask_array = np.zeros((h, w), dtype=np.uint8)
                masks.append(mask_array)
        
        return frames, masks, frame_paths
    
    def load_metadata(self, video_path: str):
        """Load captions and other metadata"""
        metadata = {}
        
        # Load complete analysis
        analysis_file = os.path.join(video_path, "complete_analysis.json")
        if os.path.exists(analysis_file):
            try:
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)
                    metadata['analysis'] = analysis
            except:
                metadata['analysis'] = {}
        
        return metadata
    
    def preprocess_frames(self, frames):
        """Preprocess frames using the enc_preprocessor"""
        if self.enc_preprocessor is not None:
            # Check what method is available in the preprocessor
            if hasattr(self.enc_preprocessor, '__call__'):
                # If it's callable, call it directly
                processed_frames = self.enc_preprocessor(frames)
            elif hasattr(self.enc_preprocessor, 'preprocess'):
                # If it has a preprocess method
                processed_frames = self.enc_preprocessor.preprocess(frames)
            elif hasattr(self.enc_preprocessor, 'process'):
                # If it has a process method
                processed_frames = self.enc_preprocessor.process(frames)
            elif hasattr(self.enc_preprocessor, 'apply_image'):
                # If it has apply_image method (like SAM)
                processed_frames = []
                for frame in frames:
                    if isinstance(frame, Image.Image):
                        frame_np = np.array(frame)
                        processed = self.enc_preprocessor.apply_image(frame_np)
                        processed_frames.append(processed)
            else:
                # If none of the above, check what methods are available
                print(f"Available methods in enc_preprocessor: {dir(self.enc_preprocessor)}")
                raise AttributeError(f"EncPreprocessor doesn't have a known preprocessing method")
        
            return processed_frames
        else:
            # Fallback preprocessing
            processed_frames = []
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_tensor = transform(frame)
                    processed_frames.append(frame_tensor)
            return processed_frames
    
    def preprocess_frames_for_sam(self, frames):
        """Preprocess frames for SAM"""
        if self.sam_preprocessor is not None:
            processed_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    # Convert PIL to numpy array
                    frame_np = np.array(frame)
                    
                    # Apply SAM preprocessing
                    if hasattr(self.sam_preprocessor, 'apply_image'):
                        # SAM v1 style
                        processed = self.sam_preprocessor.apply_image(frame_np)
                    elif hasattr(self.sam_preprocessor, '__call__'):
                        # SAM v2 style - direct call on PIL image
                        processed = self.sam_preprocessor(frame)
                    else:
                        # Fallback
                        frame_resized = frame.resize((1024, 1024))
                        processed = np.array(frame_resized).astype(np.float32) / 255.0
                    
                    processed_frames.append(processed)
            
            return processed_frames
        else:
            # Fallback: resize to SAM size
            processed_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_resized = frame.resize((1024, 1024))
                    frame_array = np.array(frame_resized).astype(np.float32) / 255.0
                    processed_frames.append(frame_array)
            return processed_frames
    
    def __getitem__(self, idx):
        video_id = self.video_dirs[idx]
        video_path = os.path.join(self.data_root, video_id)
        
        # Load frames and masks
        frames, masks, frame_paths = self.load_frames_and_masks(video_path)
        
        # Load metadata
        metadata = self.load_metadata(video_path)
        
        # Check if we have valid masks
        has_valid_mask = any(mask.max() > 0 for mask in masks)
        
        # Create conversation based on available data
        if has_valid_mask:
            # Grounding task with segmentation
            questions = [
                "Can you segment and describe the main objects in this video?",
                "Please identify and segment all moving objects in the video.",
                "What are the key objects in this video and where are they located?",
            ]
            question = random.choice(questions)
            
            # Use metadata for answer if available
            if 'analysis' in metadata and metadata['analysis']:
                if 'description' in metadata['analysis']:
                    answer = f"[SEG] {metadata['analysis']['description']}"
                else:
                    answer = "[SEG] The video shows tracked objects and their movements throughout the frames."
            else:
                answer = "[SEG] The video contains objects that are being tracked across multiple frames."
            
            # Set labels for segmentation
            labels = ["object"]  # Single label for all tracked objects
            sampled_classes = ["object"]
        else:
            # General video understanding without segmentation
            questions = [
                "What is happening in this video?",
                "Can you describe the main activities in this video?",
                "What are the key events occurring in this video?",
            ]
            question = random.choice(questions)
            
            if 'analysis' in metadata and metadata['analysis']:
                if 'description' in metadata['analysis']:
                    answer = metadata['analysis']['description']
                else:
                    answer = "This video shows various activities and movements."
            else:
                answer = "This video captures a sequence of events over time."
            
            labels = []
            sampled_classes = []
        
        # Format conversation
        conversations = [
            {"from": "human", "value": f"<video>\n{question}"},
            {"from": "gpt", "value": answer}
        ]
        
        # Preprocess frames for encoder
        preprocessed_images = self.preprocess_frames(frames)
        
        # Preprocess frames for SAM
        preprocessed_for_sam = self.preprocess_frames_for_sam(frames)
        
        # Get original size of frames
        if frames and isinstance(frames[0], Image.Image):
            orig_width, orig_height = frames[0].size
        else:
            orig_width, orig_height = 1024, 1024  # Default size
        
        # Process masks if available
        if has_valid_mask:
            # Stack masks: [num_objects, num_frames, H, W]
            # For single object tracking, num_objects = 1
            mask_tensor = np.stack(masks, axis=0)  # [num_frames, H, W]
            mask_tensor = np.expand_dims(mask_tensor, axis=0)  # [1, num_frames, H, W]
        else:
            # No masks
            mask_tensor = None
        
        # Prepare the return dictionary in the expected format
        sample = {
            'file_path': frame_paths[0] if frame_paths else "",
            'conversations': conversations,
            'label': labels,
            'resize': (orig_width, orig_height),
            'questions': question,  # The actual question asked
            'sampled_classes': sampled_classes,
            
            # Preprocessed data
            'preprocessed_for_sam': preprocessed_for_sam,  # List of preprocessed frames for SAM
            'images': preprocessed_images,  # List of preprocessed frames for encoder
            'context_images': None,  # No context images for this dataset
            'masks': mask_tensor,  # [num_objects, num_frames, H, W] or None
        }
        
        return sample