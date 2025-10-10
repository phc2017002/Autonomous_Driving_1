import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class SAMSegmentationFromBoxes:
    def __init__(self, sam_checkpoint: str = "sam_vit_h_4b8939.pth", 
                 model_type: str = "vit_h",
                 device: str = "cuda"):
        """
        Initialize SAM for segmentation from bounding boxes
        
        Args:
            sam_checkpoint: Path to SAM checkpoint
            model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        
        # Load SAM model
        print(f"Loading SAM model from {sam_checkpoint}...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        print("SAM model loaded successfully!")
        
    def load_detection_results(self, detection_file: str) -> Optional[List[Dict]]:
        """Load detection results from JSON file"""
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
                
            # Handle different possible formats
            if isinstance(data, list):
                # Direct list of detections
                return data
            elif isinstance(data, dict):
                # If it's wrapped in a dict with 'objects' key
                if 'objects' in data:
                    return data['objects']
                # If it has other structure, try to extract detections
                elif 'detections' in data:
                    return data['detections']
            
            # If empty or no valid format
            return []
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  Warning: Could not load {detection_file}: {e}")
            return []
    
    def generate_masks_for_frame(self, image_path: str, detections: List[Dict]) -> Dict:
        """
        Generate segmentation masks for all detected objects in a frame
        
        Args:
            image_path: Path to the image
            detections: List of detection dictionaries with bbox_2d
            
        Returns:
            Dictionary containing masks and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Could not load image from {image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM
        self.predictor.set_image(image_rgb)
        
        masks_data = []
        
        for idx, detection in enumerate(detections):
            # Check if bbox_2d exists
            if "bbox_2d" not in detection:
                print(f"  Warning: No bbox_2d in detection {idx}, skipping")
                continue
                
            bbox = detection["bbox_2d"]
            
            # Validate bbox
            if not bbox or len(bbox) != 4:
                print(f"  Warning: Invalid bbox for detection {idx}, skipping")
                continue
            
            # Convert bbox to SAM format (xyxy)
            input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            
            try:
                # Generate mask using bounding box
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,  # Single best mask
                )
                
                # Get the best mask
                mask = masks[0]  # Shape: (H, W)
                score = scores[0]
                
                mask_data = {
                    "object_id": idx,
                    "bbox": bbox,
                    "mask": mask.astype(np.uint8),
                    "score": float(score),
                    "object_label": detection.get("object_label", "unknown"),
                    "type": detection.get("type", "unknown"),
                    "description": detection.get("description", "")
                }
                
                masks_data.append(mask_data)
                
            except Exception as e:
                print(f"    Error generating mask for object {idx}: {e}")
                continue
        
        return {
            "image_path": image_path,
            "image_shape": image.shape,
            "num_objects": len(masks_data),
            "masks": masks_data
        }
    
    def save_masks(self, masks_result: Dict, output_dir: str, frame_name: str):
        """Save segmentation masks as images and JSON metadata"""
        if not masks_result or masks_result["num_objects"] == 0:
            print(f"  No masks to save for {frame_name}")
            return None
            
        output_path = Path(output_dir)
        
        # Create directories
        masks_dir = output_path / "masks"
        overlay_dir = output_path / "overlays"
        metadata_dir = output_path / "metadata"
        
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original image for overlays
        image = cv2.imread(masks_result["image_path"])
        
        # Create combined mask visualization
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        overlay_image = image.copy()
        
        # Generate unique colors for each object
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(masks_result["masks"]), 3))
        
        # Extract frame number from frame name
        frame_base = frame_name.replace('.jpg', '').replace('.png', '')
        
        metadata = {
            "frame": frame_name,
            "frame_number": frame_base,
            "image_path": masks_result["image_path"],
            "image_shape": masks_result["image_shape"],
            "num_objects": masks_result["num_objects"],
            "objects": []
        }
        
        for mask_data in masks_result["masks"]:
            obj_id = mask_data["object_id"]
            mask = mask_data["mask"]
            
            # Save individual mask
            mask_filename = f"{frame_base}_{obj_id:03d}_{mask_data['type'].replace(' ', '_')}.png"
            mask_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask * 255)
            
            # Add to combined mask (with different values for each object)
            combined_mask[mask > 0] = obj_id + 1
            
            # Create overlay
            color = colors[obj_id].tolist()
            overlay_mask = np.zeros_like(image)
            overlay_mask[mask > 0] = color
            overlay_image = cv2.addWeighted(overlay_image, 1.0, overlay_mask, 0.5, 0)
            
            # Add bounding box to overlay
            bbox = mask_data["bbox"]
            cv2.rectangle(overlay_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Add label
            label = f"{mask_data['type']} ({mask_data['score']:.2f})"
            cv2.putText(overlay_image, label, 
                       (int(bbox[0]), int(bbox[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add to metadata
            metadata["objects"].append({
                "object_id": obj_id,
                "bbox": mask_data["bbox"],
                "mask_file": str(mask_filename),
                "score": mask_data["score"],
                "object_label": mask_data["object_label"],
                "type": mask_data["type"],
                "description": mask_data["description"],
                "mask_area": int(np.sum(mask > 0))
            })
        
        # Save combined mask
        combined_mask_path = masks_dir / f"{frame_base}_combined.png"
        cv2.imwrite(str(combined_mask_path), combined_mask)
        
        # Save overlay image
        overlay_path = overlay_dir / f"{frame_base}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        # Save metadata
        metadata_path = metadata_dir / f"{frame_base}_masks.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def process_detection_file(self, detection_file: str, image_dir: str, output_dir: str):
        """Process a single detection JSON file and generate masks"""
        # Load detection results
        detections = self.load_detection_results(detection_file)
        
        if not detections:
            return None
        
        # Extract video name and frame name from path
        detection_path = Path(detection_file)
        
        # Expected path structure: detections_output_all_frames/VIDEO_ID/frame_detections/FRAME_NUMBER_detection.json
        frame_name_with_detection = detection_path.stem  # e.g., "000006_detection"
        frame_number = frame_name_with_detection.replace('_detection', '')  # "000006"
        frame_name = frame_number + '.jpg'  # "000006.jpg"
        
        # Get video name from parent directories
        video_name = None
        if 'frame_detections' in detection_path.parts:
            idx = detection_path.parts.index('frame_detections')
            if idx > 0:
                video_name = detection_path.parts[idx - 1]  # This should be "002320"
        
        if not video_name:
            video_name = detection_path.parent.parent.name
        
        # Find image file - try multiple possible locations
        possible_paths = [
            # Most likely paths based on common structures
            Path(image_dir) / video_name / "images" / frame_name,
            Path(image_dir) / video_name / frame_name,
            Path(image_dir) / "images" / video_name / frame_name,
            Path(image_dir) / f"video_{video_name}" / frame_name,
            Path(image_dir) / f"video_{video_name}" / "images" / frame_name,
            
            # Also try with frame number directly (without extension)
            Path(image_dir) / video_name / "images" / frame_number / "image.jpg",
            Path(image_dir) / video_name / "images" / frame_number / "frame.jpg",
            
            # Try PNG format
            Path(image_dir) / video_name / "images" / frame_name.replace('.jpg', '.png'),
            Path(image_dir) / video_name / frame_name.replace('.jpg', '.png'),
            
            # Direct path
            Path(image_dir) / frame_name,
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = str(path)
                break
        
        if not image_path:
            print(f"  Warning: Could not find image for {video_name}/{frame_name}")
            return None
        
        # Generate masks
        masks_result = self.generate_masks_for_frame(image_path, detections)
        
        if not masks_result:
            return None
        
        # Save masks
        output_path = Path(output_dir) / video_name
        metadata = self.save_masks(masks_result, str(output_path), frame_name)
        
        return metadata
    
    def process_video_detections(self, video_dir: str, image_dir: str, output_dir: str, 
                                max_frames: int = None):
        """Process all detection files for a specific video"""
        video_path = Path(video_dir)
        video_name = video_path.name  # "002320"
        
        # Find all detection JSON files in frame_detections folder
        frame_detections_dir = video_path / "frame_detections"
        if not frame_detections_dir.exists():
            print(f"Error: {frame_detections_dir} does not exist")
            return []
        
        detection_files = sorted(list(frame_detections_dir.glob("*_detection.json")))
        
        if max_frames:
            detection_files = detection_files[:max_frames]
        
        print(f"Found {len(detection_files)} detection files for video {video_name}")
        
        all_results = []
        failed_files = []
        
        # Use tqdm for progress bar
        for detection_file in tqdm(detection_files, desc=f"Processing {video_name}"):
            try:
                result = self.process_detection_file(str(detection_file), image_dir, output_dir)
                if result:
                    all_results.append(result)
                else:
                    failed_files.append(str(detection_file))
            except Exception as e:
                print(f"\n  Error processing {detection_file.name}: {e}")
                failed_files.append(str(detection_file))
        
        # Save video-specific summary
        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = video_output_dir / "segmentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "video_id": video_name,
                "total_frames_processed": len(all_results),
                "total_frames_failed": len(failed_files),
                "failed_files": failed_files,
                "frames_with_masks": [r["frame_number"] for r in all_results if r]
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Video {video_name} Results:")
        print(f"  ✅ Successfully processed: {len(all_results)}/{len(detection_files)} frames")
        if failed_files:
            print(f"  ⚠️  Failed: {len(failed_files)} frames")
        print(f"  📁 Results saved to: {video_output_dir}")
        
        return all_results
    
    def process_all_videos(self, detections_base_dir: str, image_dir: str, output_dir: str, 
                          video_ids: List[str] = None, max_frames_per_video: int = None):
        """Process detection files for multiple videos"""
        detections_path = Path(detections_base_dir)
        
        # If specific video IDs provided, use those; otherwise find all
        if video_ids:
            video_dirs = [detections_path / vid for vid in video_ids if (detections_path / vid).exists()]
        else:
            video_dirs = [d for d in detections_path.iterdir() if d.is_dir()]
        
        print(f"Processing {len(video_dirs)} videos")
        
        all_video_results = {}
        
        for video_dir in video_dirs:
            video_name = video_dir.name
            print(f"\n{'='*60}")
            print(f"Processing video: {video_name}")
            print(f"{'='*60}")
            
            results = self.process_video_detections(
                str(video_dir), 
                image_dir, 
                output_dir,
                max_frames=max_frames_per_video
            )
            
            all_video_results[video_name] = {
                "num_frames": len(results),
                "results": results
            }
        
        # Save overall summary
        summary_path = Path(output_dir) / "overall_segmentation_summary.json"
        with open(summary_path, 'w') as f:
            total_frames = sum(v["num_frames"] for v in all_video_results.values())
            json.dump({
                "total_videos": len(all_video_results),
                "total_frames_processed": total_frames,
                "videos": {k: v["num_frames"] for k, v in all_video_results.items()}
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"🎉 All Processing Complete!")
        print(f"  📹 Videos processed: {len(all_video_results)}")
        print(f"  🖼️  Total frames: {total_frames}")
        print(f"  📁 Results saved to: {output_dir}")
        
        return all_video_results


def main():
    # Configuration for your specific setup
    BASE_DIR = "/data/users/apal/CAP-DATA_44-62/VideoGLaMM/VideoGLaMM"
    
    # Initialize SAM segmenter
    segmenter = SAMSegmentationFromBoxes(
        sam_checkpoint=f"{BASE_DIR}/sam_vit_h_4b8939.pth",  # Adjust path to your SAM checkpoint
        model_type="vit_h",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Directory containing your detection JSON files
    detections_dir = f"{BASE_DIR}/detections_output_all_frames"
    
    # Directory containing the actual images
    # You need to specify where your video frames/images are stored
    # Common structures:
    # - BASE_DIR/videos/002320/images/000001.jpg
    # - BASE_DIR/frames/002320/000001.jpg
    # - BASE_DIR/002320/images/000001.jpg
    image_dir = f"{BASE_DIR}/002320/images"  # Adjust this to your actual image directory
    
    # Output directory for segmentation results
    output_dir = f"{BASE_DIR}/segmentation_output"
    
    # Process video 002320 (all 113 frames)
    segmenter.process_video_detections(
        video_dir=f"{detections_dir}/002320",
        image_dir=image_dir,
        output_dir=output_dir,
        max_frames=None  # Process all 113 frames, or set a number for testing
    )
    
    # Or process all videos in the detections directory
    # segmenter.process_all_videos(
    #     detections_base_dir=detections_dir,
    #     image_dir=image_dir,
    #     output_dir=output_dir,
    #     video_ids=["002320"],  # Specify video IDs or None for all
    #     max_frames_per_video=10  # Limit frames per video for testing
    # )


if __name__ == "__main__":
    main()