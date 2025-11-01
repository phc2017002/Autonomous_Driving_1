import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

class MMAUDataset(Dataset):
    def __init__(
        self,
        base_dir,
        tokenizer,
        vision_tower,
        sam_preprocessor,
        enc_preprocessor,
        conversation_generator,
        image_size=1024,
        num_frames_per_video=8,
        split='train',
    ):
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.sam_preprocessor = sam_preprocessor
        self.enc_preprocessor = enc_preprocessor
        self.conversation_generator = conversation_generator
        self.image_size = image_size
        self.num_frames = num_frames_per_video
        self.split = split
        
        # Load dataset annotations
        self.samples = self._load_annotations()
        
    def _load_annotations(self):
        """Load MM-AU dataset structure"""
        samples = []
        
        # Assuming structure: base_dir/video_folders/
        for video_folder in os.listdir(self.base_dir):
            video_path = os.path.join(self.base_dir, video_folder)
            if not os.path.isdir(video_path):
                continue
                
            # Load grounded caption (assuming JSON format)
            caption_file = os.path.join(video_path, 'grounded_caption.json')
            if os.path.exists(caption_file):
                with open(caption_file, 'r') as f:
                    caption_data = json.load(f)
                
                # Get frame files
                frames = sorted([f for f in os.listdir(video_path) 
                               if f.endswith(('.jpg', '.png')) and 'mask' not in f])
                
                # Get mask files (accident overlays)
                masks = sorted([f for f in os.listdir(video_path) 
                              if 'mask' in f or 'overlay' in f])
                
                samples.append({
                    'video_id': video_folder,
                    'frames': frames,
                    'masks': masks,
                    'caption_data': caption_data,
                    'video_path': video_path
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Sample frames uniformly
        frame_indices = self._sample_frames(len(sample['frames']))
        
        # Load frames
        frames = []
        masks = []
        
        for i in frame_indices:
            # Load frame
            frame_path = os.path.join(sample['video_path'], sample['frames'][i])
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
            
            # Load corresponding mask if exists
            if i < len(sample['masks']):
                mask_path = os.path.join(sample['video_path'], sample['masks'][i])
                mask = Image.open(mask_path).convert('L')  # Grayscale mask
            else:
                # Create empty mask if not available
                mask = Image.new('L', frame.size, 0)
            masks.append(mask)
        
        # Process for video encoder
        video_clip = self.enc_preprocessor(frames)  # Process frames for video encoder
        
        # Process masks for SAM
        sam_masks = []
        for mask in masks:
            mask_array = np.array(mask.resize((self.image_size, self.image_size)))
            mask_tensor = torch.from_numpy(mask_array).float() / 255.0
            sam_masks.append(mask_tensor)
        sam_masks = torch.stack(sam_masks)
        
        # Generate conversation for accident grounding task
        conversation = self._generate_conversation(sample['caption_data'])
        
        return {
            'video_clip': video_clip,
            'masks': sam_masks,
            'conversation': conversation,
            'video_id': sample['video_id']
        }
    
    def _sample_frames(self, total_frames):
        """Sample frames uniformly from video"""
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return indices.tolist()
    
    def _generate_conversation(self, caption_data):
        """Generate conversation for accident grounding"""
        # Example conversation templates for accident detection
        templates = [
            "Identify and segment the accident objects in this video sequence.",
            "What accident-related objects can you detect? Please segment them.",
            "Locate and segment any dangerous or accident-prone objects in these frames.",
            f"Based on the scene: {caption_data.get('scene_description', '')}, identify accident objects.",
        ]
        
        question = random.choice(templates)
        
        # Format answer with grounding
        answer = f"[SEG] The accident objects detected are: {caption_data.get('objects', 'various accident-related items')}. "
        answer += f"Context: {caption_data.get('grounded_caption', '')}"
        
        return {
            'question': question,
            'answer': answer,
            'has_mask': True
        }