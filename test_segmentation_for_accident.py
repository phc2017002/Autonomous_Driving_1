import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

class AccidentSAMSegmentation:
    def __init__(self, sam_checkpoint: str = "sam_vit_h_4b8939.pth", 
                 model_type: str = "vit_h",
                 device: str = "cuda"):
        """
        Initialize SAM for accident object segmentation from bounding boxes
        
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
        
        # Define color schemes for different object types
        self.accident_colors = {
            "accident_object_1": (255, 0, 0),      # Red - Primary accident object (e.g., detached tire)
            "accident_object_2": (255, 127, 0),    # Orange - Damaged vehicle
            "accident_object_3": (255, 255, 0),    # Yellow - Reacting vehicles
            "accident_object_4": (255, 0, 255),    # Magenta - Debris
        }
        
        self.normal_object_colors = {
            "vehicle": (0, 255, 0),                # Green
            "person": (0, 0, 255),                 # Blue
            "infrastructure": (128, 128, 128),     # Gray
            "road_element": (64, 64, 64),          # Dark gray
            "environment": (0, 128, 0),            # Dark green
            "traffic_sign": (255, 255, 0),         # Yellow
            "barrier": (128, 64, 0),               # Brown
        }
    
    def load_accident_summary(self, summary_file: str) -> Dict:
        """Load accident summary JSON file"""
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {summary_file}: {e}")
            return None
    
    def extract_frame_detections(self, accident_summary: Dict) -> Dict[str, List[Dict]]:
        """
        Extract detections organized by frame from accident summary
        
        Returns:
            Dictionary mapping frame names to their detections
        """
        frame_detections = {}
        
        # Check if frame_results exists
        if "frame_results" in accident_summary:
            for frame_result in accident_summary["frame_results"]:
                frame_name = frame_result.get("frame")
                if frame_name and "objects" in frame_result:
                    frame_detections[frame_name] = frame_result["objects"]
        
        # Alternative: check if detected_objects exists (flat list)
        elif "detected_objects" in accident_summary:
            for obj in accident_summary["detected_objects"]:
                frame_name = obj.get("frame")
                if frame_name:
                    if frame_name not in frame_detections:
                        frame_detections[frame_name] = []
                    frame_detections[frame_name].append(obj)
        
        return frame_detections
    
    def get_object_color(self, detection: Dict) -> Tuple[int, int, int]:
        """Get color for object based on its type and accident relation"""
        object_label = detection.get("object_label", "")
        
        # Check if it's an accident object
        if detection.get("is_accident_related", False) or "accident_object" in object_label:
            # Try to get specific accident object color
            if object_label in self.accident_colors:
                return self.accident_colors[object_label]
            # Default accident color
            return (255, 0, 128)  # Bright pink for unspecified accident objects
        
        # Normal object colors
        return self.normal_object_colors.get(object_label, (128, 128, 128))
    
    def generate_masks_for_frame(self, image_path: str, detections: List[Dict], 
                                 frame_info: Dict = None) -> Dict:
        """
        Generate segmentation masks for all detected objects in a frame
        
        Args:
            image_path: Path to the image
            detections: List of detection dictionaries with bbox_2d
            frame_info: Additional frame information (frame number, accident frame, etc.)
            
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
        accident_masks = []
        normal_masks = []
        
        for idx, detection in enumerate(detections):
            # Check if bbox_2d exists
            if "bbox_2d" not in detection:
                continue
                
            bbox = detection["bbox_2d"]
            
            # Validate bbox
            if not bbox or len(bbox) != 4:
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
                
                # Determine if accident-related
                is_accident = detection.get("is_accident_related", False) or \
                            "accident_object" in detection.get("object_label", "")
                
                mask_data = {
                    "object_id": idx,
                    "bbox": bbox,
                    "mask": mask.astype(np.uint8),
                    "score": float(score),
                    "object_label": detection.get("object_label", "unknown"),
                    "type": detection.get("type", "unknown"),
                    "description": detection.get("description", ""),
                    "is_accident_related": is_accident,
                    "frame_number": detection.get("frame_number"),
                    "frames_from_accident": detection.get("frames_from_accident")
                }
                
                masks_data.append(mask_data)
                
                if is_accident:
                    accident_masks.append(mask_data)
                else:
                    normal_masks.append(mask_data)
                
            except Exception as e:
                print(f"    Error generating mask for object {idx}: {e}")
                continue
        
        return {
            "image_path": image_path,
            "image_shape": image.shape,
            "num_objects": len(masks_data),
            "num_accident_objects": len(accident_masks),
            "num_normal_objects": len(normal_masks),
            "masks": masks_data,
            "accident_masks": accident_masks,
            "normal_masks": normal_masks,
            "frame_info": frame_info
        }
    
    def save_accident_masks(self, masks_result: Dict, output_dir: str, frame_name: str):
        """Save segmentation masks with special handling for accident objects"""
        if not masks_result or masks_result["num_objects"] == 0:
            print(f"  No masks to save for {frame_name}")
            return None
            
        output_path = Path(output_dir)
        
        # Create directories
        masks_dir = output_path / "masks"
        accident_masks_dir = output_path / "accident_masks"
        overlay_dir = output_path / "overlays"
        accident_overlay_dir = output_path / "accident_overlays"
        metadata_dir = output_path / "metadata"
        
        for dir_path in [masks_dir, accident_masks_dir, overlay_dir, 
                        accident_overlay_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load original image
        image = cv2.imread(masks_result["image_path"])
        
        # Create visualizations
        overlay_all = image.copy()
        overlay_accident_only = image.copy()
        combined_mask_all = np.zeros(image.shape[:2], dtype=np.uint8)
        combined_mask_accident = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Extract frame number
        frame_base = frame_name.replace('.jpg', '').replace('.png', '')
        
        metadata = {
            "frame": frame_name,
            "frame_number": frame_base,
            "image_path": masks_result["image_path"],
            "image_shape": masks_result["image_shape"],
            "num_objects": masks_result["num_objects"],
            "num_accident_objects": masks_result["num_accident_objects"],
            "num_normal_objects": masks_result["num_normal_objects"],
            "frame_info": masks_result.get("frame_info"),
            "objects": [],
            "accident_objects": []
        }
        
        accident_obj_counter = 0
        
        for mask_data in masks_result["masks"]:
            obj_id = mask_data["object_id"]
            mask = mask_data["mask"]
            is_accident = mask_data["is_accident_related"]
            
            # Get appropriate color
            if is_accident:
                # Use accident-specific colors
                object_label = mask_data["object_label"]
                if object_label in self.accident_colors:
                    color = self.accident_colors[object_label]
                else:
                    color = (255, 0, 128)  # Default accident color
            else:
                # Use normal object colors
                object_label = mask_data.get("object_label", "unknown")
                color = self.normal_object_colors.get(object_label, (128, 128, 128))
            
            # Save individual mask
            if is_accident:
                mask_filename = f"{frame_base}_accident_{accident_obj_counter:02d}_{mask_data['object_label']}.png"
                mask_path = accident_masks_dir / mask_filename
                accident_obj_counter += 1
            else:
                mask_filename = f"{frame_base}_{obj_id:03d}_{mask_data['type'].replace(' ', '_')}.png"
                mask_path = masks_dir / mask_filename
            
            cv2.imwrite(str(mask_path), mask * 255)
            
            # Add to combined masks
            combined_mask_all[mask > 0] = obj_id + 1
            if is_accident:
                combined_mask_accident[mask > 0] = accident_obj_counter
            
            # Create overlay for all objects
            overlay_mask = np.zeros_like(image)
            overlay_mask[mask > 0] = color
            overlay_all = cv2.addWeighted(overlay_all, 1.0, overlay_mask, 0.5, 0)
            
            # Create overlay for accident objects only
            if is_accident:
                overlay_accident_only = cv2.addWeighted(overlay_accident_only, 1.0, overlay_mask, 0.6, 0)
            
            # Add bounding boxes and labels
            bbox = mask_data["bbox"]
            thickness = 3 if is_accident else 2
            
            # Draw on all objects overlay
            cv2.rectangle(overlay_all, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, thickness)
            
            # Add label with frame info if available
            label = f"{mask_data['type']}"
            if mask_data.get("frames_from_accident") is not None:
                label += f" [{mask_data['frames_from_accident']:+d}]"
            label += f" ({mask_data['score']:.2f})"
            
            font_scale = 0.6 if is_accident else 0.5
            font_thickness = 2 if is_accident else 1
            
            cv2.putText(overlay_all, label, 
                       (int(bbox[0]), int(bbox[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            
            # Draw on accident-only overlay if accident object
            if is_accident:
                cv2.rectangle(overlay_accident_only, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), 
                             color, thickness)
                cv2.putText(overlay_accident_only, label, 
                           (int(bbox[0]), int(bbox[1] - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            
            # Add to metadata
            obj_metadata = {
                "object_id": obj_id,
                "bbox": mask_data["bbox"],
                "mask_file": str(mask_filename),
                "score": mask_data["score"],
                "object_label": mask_data["object_label"],
                "type": mask_data["type"],
                "description": mask_data["description"],
                "is_accident_related": is_accident,
                "mask_area": int(np.sum(mask > 0)),
                "frame_number": mask_data.get("frame_number"),
                "frames_from_accident": mask_data.get("frames_from_accident")
            }
            
            metadata["objects"].append(obj_metadata)
            if is_accident:
                metadata["accident_objects"].append(obj_metadata)
        
        # Save combined masks
        combined_all_path = masks_dir / f"{frame_base}_combined_all.png"
        cv2.imwrite(str(combined_all_path), combined_mask_all)
        
        if accident_obj_counter > 0:
            combined_accident_path = accident_masks_dir / f"{frame_base}_combined_accident.png"
            cv2.imwrite(str(combined_accident_path), combined_mask_accident)
        
        # Save overlay images
        overlay_all_path = overlay_dir / f"{frame_base}_overlay_all.jpg"
        cv2.imwrite(str(overlay_all_path), overlay_all)
        
        if accident_obj_counter > 0:
            overlay_accident_path = accident_overlay_dir / f"{frame_base}_overlay_accident.jpg"
            cv2.imwrite(str(overlay_accident_path), overlay_accident_only)
        
        # Add frame info text to overlays
        if masks_result.get("frame_info"):
            frame_info = masks_result["frame_info"]
            info_text = f"Frame {frame_info.get('frame_number', 'N/A')}"
            if frame_info.get('frames_from_accident') is not None:
                info_text += f" | Accident {frame_info['frames_from_accident']:+d}"
            
            cv2.putText(overlay_all, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if accident_obj_counter > 0:
                cv2.putText(overlay_accident_only, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save metadata
        metadata_path = metadata_dir / f"{frame_base}_masks.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def process_accident_video(self, video_id: str, accident_dir: str, image_dir: str, 
                              output_dir: str, max_frames: int = None):
        """
        Process accident detection results for a video
        
        Args:
            video_id: Video identifier (e.g., "002320")
            accident_dir: Directory containing accident detection results
            image_dir: Directory containing video frames
            output_dir: Output directory for segmentation results
            max_frames: Maximum number of frames to process
        """
        # Load accident summary
        summary_file = Path(accident_dir) / video_id / "accident_summary.json"
        if not summary_file.exists():
            print(f"Error: Accident summary not found at {summary_file}")
            return None
        
        accident_summary = self.load_accident_summary(str(summary_file))
        if not accident_summary:
            return None
        
        # Extract metadata
        metadata = accident_summary.get("metadata", {})
        accident_frame = metadata.get("accident_frame")
        
        print(f"\n{'='*60}")
        print(f"Processing accident video: {video_id}")
        print(f"  Accident frame: {accident_frame}")
        print(f"  Cause: {metadata.get('cause', 'Unknown')}")
        print(f"{'='*60}")
        
        # Extract frame detections
        frame_detections = self.extract_frame_detections(accident_summary)
        
        if not frame_detections:
            print("No frame detections found in summary")
            return None
        
        # Sort frames by frame number
        sorted_frames = sorted(frame_detections.keys(), 
                             key=lambda x: int(x.replace('.jpg', '').replace('000', '')))
        
        if max_frames:
            sorted_frames = sorted_frames[:max_frames]
        
        print(f"Processing {len(sorted_frames)} frames with detections...")
        
        # Create output directory
        video_output_dir = Path(output_dir) / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        accident_frame_results = []
        
        # Process each frame
        for frame_name in tqdm(sorted_frames, desc=f"Generating masks for {video_id}"):
            detections = frame_detections[frame_name]
            
            if not detections:
                continue
            
            # Extract frame number
            frame_number = int(frame_name.replace('.jpg', '').replace('000', ''))
            
            # Calculate frames from accident
            frames_from_accident = None
            if accident_frame:
                frames_from_accident = frame_number - int(accident_frame)
            
            # Build frame info
            frame_info = {
                "frame_number": frame_number,
                "accident_frame": accident_frame,
                "frames_from_accident": frames_from_accident
            }
            
            # Find image file
            possible_paths = [
                Path(image_dir) / video_id / "images" / frame_name,
                Path(image_dir) / video_id / frame_name,
                Path(image_dir) / frame_name,
                # Direct path if image_dir already points to the specific video
                Path(image_dir) / frame_name,
            ]
            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = str(path)
                    break
            
            if not image_path:
                print(f"  Warning: Image not found for {frame_name}")
                continue
            
            # Generate masks
            masks_result = self.generate_masks_for_frame(image_path, detections, frame_info)
            
            if not masks_result:
                continue
            
            # Save masks with accident highlighting
            metadata = self.save_accident_masks(masks_result, str(video_output_dir), frame_name)
            
            if metadata:
                all_results.append(metadata)
                
                # Track accident frames specifically
                if frames_from_accident is not None and abs(frames_from_accident) <= 20:
                    accident_frame_results.append({
                        "frame": frame_name,
                        "frame_number": frame_number,
                        "frames_from_accident": frames_from_accident,
                        "num_accident_objects": metadata["num_accident_objects"],
                        "accident_objects": metadata["accident_objects"]
                    })
        
        # Save video segmentation summary
        summary_path = video_output_dir / "segmentation_summary.json"
        
        # Count total accident objects across all frames
        total_accident_objects = sum(r["num_accident_objects"] for r in all_results)
        total_normal_objects = sum(r["num_normal_objects"] for r in all_results)
        
        with open(summary_path, 'w') as f:
            json.dump({
                "video_id": video_id,
                "accident_frame": accident_frame,
                "cause": metadata.get("cause"),
                "total_frames_processed": len(all_results),
                "total_objects_segmented": total_accident_objects + total_normal_objects,
                "total_accident_objects": total_accident_objects,
                "total_normal_objects": total_normal_objects,
                "accident_sequence_frames": accident_frame_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Segmentation complete for {video_id}")
        print(f"  Frames processed: {len(all_results)}")
        print(f"  Total accident objects segmented: {total_accident_objects}")
        print(f"  Total normal objects segmented: {total_normal_objects}")
        print(f"  Results saved to: {video_output_dir}")
        print(f"{'='*60}\n")
        
        return all_results
    
    def create_accident_sequence_video(self, video_id: str, segmentation_dir: str, 
                                      output_video_path: str = None, fps: int = 10):
        """
        Create a video from the accident sequence overlays
        
        Args:
            video_id: Video identifier
            segmentation_dir: Directory containing segmentation results
            output_video_path: Path for output video (optional)
            fps: Frames per second for the video
        """
        video_seg_dir = Path(segmentation_dir) / video_id
        accident_overlay_dir = video_seg_dir / "accident_overlays"
        
        if not accident_overlay_dir.exists():
            print(f"No accident overlays found for {video_id}")
            return None
        
        # Get all overlay images
        overlay_files = sorted(list(accident_overlay_dir.glob("*_overlay_accident.jpg")))
        
        if not overlay_files:
            print(f"No overlay files found in {accident_overlay_dir}")
            return None
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(overlay_files[0]))
        height, width, layers = first_img.shape
        
        # Set output path
        if not output_video_path:
            output_video_path = video_seg_dir / f"{video_id}_accident_sequence.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        print(f"Creating accident sequence video with {len(overlay_files)} frames...")
        
        for overlay_file in tqdm(overlay_files, desc="Writing video"):
            frame = cv2.imread(str(overlay_file))
            video_writer.write(frame)
        
        video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"Video saved to: {output_video_path}")
        return str(output_video_path)


def main():
    # Configuration
    BASE_DIR = "."  # Adjust to your directory
    
    # Initialize SAM segmenter for accident detection
    segmenter = AccidentSAMSegmentation(
        sam_checkpoint=f"{BASE_DIR}/sam_vit_h_4b8939.pth",  # Adjust path
        model_type="vit_h",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Process accident detection results
    video_id = "002320"
    
    # Directory containing accident detection results
    accident_dir = f"{BASE_DIR}/accident_detections"
    
    # Directory containing the actual images
    # This should point to where your frames are stored
    image_dir = f"{BASE_DIR}/002320/images"  # Adjust based on your structure
    
    # Output directory for segmentation results
    output_dir = f"{BASE_DIR}/accident_segmentation_output"
    
    # Process the accident video
    results = segmenter.process_accident_video(
        video_id=video_id,
        accident_dir=accident_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        max_frames=None  # Process all frames, or set a limit for testing
    )
    
    # Optional: Create a video from the accident sequence
    if results:
        segmenter.create_accident_sequence_video(
            video_id=video_id,
            segmentation_dir=output_dir,
            fps=10
        )
    
    print("\n" + "="*60)
    print("🎉 Processing Complete!")
    print(f"Check the following directories:")
    print(f"  📁 {output_dir}/{video_id}/masks/ - Individual object masks")
    print(f"  📁 {output_dir}/{video_id}/accident_masks/ - Accident object masks")
    print(f"  📁 {output_dir}/{video_id}/overlays/ - All objects visualization")
    print(f"  📁 {output_dir}/{video_id}/accident_overlays/ - Accident objects only")
    print(f"  📄 {output_dir}/{video_id}/segmentation_summary.json - Summary statistics")
    print("="*60)


if __name__ == "__main__":
    main()