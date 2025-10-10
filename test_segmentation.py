import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt

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
        
    def load_detection_results(self, detection_file: str) -> Dict:
        """Load detection results from JSON file"""
        with open(detection_file, 'r') as f:
            return json.load(f)
    
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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM
        self.predictor.set_image(image_rgb)
        
        masks_data = []
        
        for idx, detection in enumerate(detections):
            bbox = detection["bbox_2d"]
            
            # Convert bbox to SAM format (xyxy)
            input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            
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
            
            print(f"    Generated mask for object {idx}: {detection['type']} "
                  f"({detection['description']}) - Score: {score:.3f}")
        
        return {
            "image_path": image_path,
            "image_shape": image.shape,
            "num_objects": len(masks_data),
            "masks": masks_data
        }
    
    def save_masks(self, masks_result: Dict, output_dir: str, frame_name: str):
        """Save segmentation masks as images and JSON metadata"""
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
        
        metadata = {
            "frame": frame_name,
            "image_path": masks_result["image_path"],
            "image_shape": masks_result["image_shape"],
            "num_objects": masks_result["num_objects"],
            "objects": []
        }
        
        for mask_data in masks_result["masks"]:
            obj_id = mask_data["object_id"]
            mask = mask_data["mask"]
            
            # Save individual mask
            mask_filename = f"{frame_name.replace('.jpg', '')}_{obj_id:03d}_{mask_data['type']}.png"
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
        combined_mask_path = masks_dir / f"{frame_name.replace('.jpg', '')}_combined.png"
        cv2.imwrite(str(combined_mask_path), combined_mask)
        
        # Save overlay image
        overlay_path = overlay_dir / f"{frame_name.replace('.jpg', '')}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        # Save metadata
        metadata_path = metadata_dir / f"{frame_name.replace('.jpg', '')}_masks.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved masks to {masks_dir}")
        print(f"  Saved overlay to {overlay_path}")
        
        return metadata
    
    def process_detection_file(self, detection_file: str, image_dir: str, output_dir: str):
        """Process a single detection JSON file and generate masks"""
        # Load detection results
        detection_data = self.load_detection_results(detection_file)
        
        video_name = detection_data["video"]
        frame_name = detection_data["frame"]
        objects = detection_data["objects"]
        
        print(f"\nProcessing {video_name}/{frame_name} with {len(objects)} objects")
        
        # Find image file
        possible_paths = [
            Path(image_dir) / video_name / "images" / frame_name,
            Path(image_dir) / video_name / frame_name,
            Path(image_dir) / frame_name,
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = str(path)
                break
        
        if not image_path:
            print(f"  Error: Could not find image at any of: {possible_paths}")
            return None
        
        # Generate masks
        masks_result = self.generate_masks_for_frame(image_path, objects)
        
        # Save masks
        output_path = Path(output_dir) / video_name
        metadata = self.save_masks(masks_result, str(output_path), frame_name)
        
        return metadata
    
    def process_all_detections(self, detections_dir: str, image_dir: str, output_dir: str):
        """Process all detection files in a directory"""
        detections_path = Path(detections_dir)
        
        # Find all detection JSON files
        detection_files = list(detections_path.rglob("*_detection.json"))
        
        print(f"Found {len(detection_files)} detection files to process")
        
        all_results = []
        
        for detection_file in detection_files:
            print(f"\nProcessing: {detection_file}")
            result = self.process_detection_file(str(detection_file), image_dir, output_dir)
            if result:
                all_results.append(result)
        
        # Save summary
        summary_path = Path(output_dir) / "segmentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "total_frames_processed": len(all_results),
                "results": all_results
            }, f, indent=2)
        
        print(f"\n✅ Completed! Processed {len(all_results)} frames")
        print(f"Results saved to: {output_dir}")
        
        return all_results


# Option 2: Grounding SAM (if you want to use text descriptions)
class GroundingSAMSegmentation:
    def __init__(self, grounding_dino_checkpoint: str, sam_checkpoint: str, device: str = "cuda"):
        """
        Initialize Grounding SAM for text-prompted segmentation
        
        Note: Requires GroundingDINO and SAM models
        """
        from groundingdino.util.inference import Model as GroundingDINO
        
        self.device = device
        
        # Load Grounding DINO
        self.grounding_dino = GroundingDINO(
            model_config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path=grounding_dino_checkpoint
        )
        
        # Load SAM
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
    def generate_masks_with_text(self, image_path: str, detections: List[Dict]):
        """Generate masks using text descriptions and bounding boxes"""
        # Implementation similar to above but uszing text prompts
        # This would use Grounding DINO to refine boxes based on text
        pass


def main():
    # Example usage
    
    # Initialize SAM segmenter
    segmenter = SAMSegmentationFromBoxes(
        sam_checkpoint="sam_vit_h_4b8939.pth",  # Download from Facebook SAM repo
        model_type="vit_h",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Process single detection file
    detection_file = "detections_output/002320/frame_detections/000001_detection.json"
    image_dir = "."  # Directory containing video folders
    output_dir = "segmentation_output"
    
    # Process single file
    segmenter.process_detection_file(detection_file, image_dir, output_dir)
    
    # Or process all detections in a directory
    # segmenter.process_all_detections(
    #     detections_dir="detections_output",
    #     image_dir=".",
    #     output_dir="segmentation_output"
    # )


if __name__ == "__main__":
    main()