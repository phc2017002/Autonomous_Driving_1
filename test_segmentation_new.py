import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt

class SAMSegmentationFromBoxes:
    def __init__(self, sam_checkpoint: str = "sam_vit_h_4b8939.pth", 
                 model_type: str = "vit_h",
                 device: str = "cuda",
                 class_names: Dict[int, str] = None):
        """
        Initialize SAM for segmentation from bounding boxes
        
        Args:
            sam_checkpoint: Path to SAM checkpoint
            model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to run on (cuda/cpu)
            class_names: Dictionary mapping class IDs to names
        """
        self.device = device
        
        # Default class names (you can customize these)
        self.class_names = class_names or {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            # Add more class names as needed
        }
        
        # Load SAM model
        print(f"Loading SAM model from {sam_checkpoint}...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        print("SAM model loaded successfully!")
    
    def yolo_to_xyxy(self, box: List[float], img_width: int, img_height: int) -> List[int]:
        """
        Convert YOLO format (x_center, y_center, width, height) normalized to xyxy pixel coordinates
        
        Args:
            box: [x_center, y_center, width, height] in normalized coordinates (0-1)
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            [x1, y1, x2, y2] in pixel coordinates
        """
        x_center, y_center, width, height = box
        
        # Convert from normalized to pixel coordinates
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Convert from center format to corner format
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        return [x1, y1, x2, y2]
    
    def load_yolo_boxes(self, bbox_file: str) -> List[Tuple[int, List[float]]]:
        """
        Load bounding boxes from YOLO format text file
        
        Args:
            bbox_file: Path to text file with YOLO format boxes
            
        Returns:
            List of tuples (class_id, [x_center, y_center, width, height])
        """
        boxes = []
        
        if not Path(bbox_file).exists():
            print(f"Warning: Bbox file not found: {bbox_file}")
            return boxes
        
        with open(bbox_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    boxes.append((class_id, bbox))
        
        return boxes
    
    def generate_masks_from_yolo_boxes(self, image_path: str, bbox_file: str) -> Dict:
        """
        Generate segmentation masks from YOLO format bounding boxes
        
        Args:
            image_path: Path to the image
            bbox_file: Path to YOLO format bbox file
            
        Returns:
            Dictionary containing masks and metadata
        """
        # Load image
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Load bounding boxes
        yolo_boxes = self.load_yolo_boxes(bbox_file)
        
        if not yolo_boxes:
            print(f"No bounding boxes found in {bbox_file}")
            return None
        
        print(f"Processing {len(yolo_boxes)} bounding boxes from {bbox_file}")
        
        # Set image for SAM
        self.predictor.set_image(image_rgb)
        
        masks_data = []
        
        for idx, (class_id, yolo_box) in enumerate(yolo_boxes):
            # Convert YOLO format to xyxy pixel coordinates
            bbox = self.yolo_to_xyxy(yolo_box, width, height)
            
            # Convert bbox to SAM format (xyxy)
            input_box = np.array(bbox)
            
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
            
            # Get class name
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            mask_data = {
                "object_id": idx,
                "class_id": class_id,
                "class_name": class_name,
                "bbox": bbox,
                "yolo_bbox": yolo_box,
                "mask": mask.astype(np.uint8),
                "score": float(score),
            }
            
            masks_data.append(mask_data)
            
            print(f"    Generated mask for object {idx}: class {class_id} ({class_name}) - Score: {score:.3f}")
        
        return {
            "image_path": image_path,
            "bbox_file": bbox_file,
            "image_shape": image.shape,
            "num_objects": len(masks_data),
            "masks": masks_data
        }
    
    def save_masks(self, masks_result: Dict, output_dir: str, base_name: str):
        """Save segmentation masks as images and JSON metadata"""
        if masks_result is None:
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
        
        metadata = {
            "base_name": base_name,
            "image_path": masks_result["image_path"],
            "bbox_file": masks_result["bbox_file"],
            "image_shape": masks_result["image_shape"],
            "num_objects": masks_result["num_objects"],
            "objects": []
        }
        
        for mask_data in masks_result["masks"]:
            obj_id = mask_data["object_id"]
            mask = mask_data["mask"]
            
            # Save individual mask
            mask_filename = f"{base_name}_{obj_id:03d}_{mask_data['class_name']}.png"
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
            label = f"{mask_data['class_name']} ({mask_data['score']:.2f})"
            cv2.putText(overlay_image, label, 
                       (int(bbox[0]), int(bbox[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add to metadata
            metadata["objects"].append({
                "object_id": obj_id,
                "class_id": mask_data["class_id"],
                "class_name": mask_data["class_name"],
                "bbox": mask_data["bbox"],
                "yolo_bbox": mask_data["yolo_bbox"],
                "mask_file": str(mask_filename),
                "score": mask_data["score"],
                "mask_area": int(np.sum(mask > 0))
            })
        
        # Save combined mask
        combined_mask_path = masks_dir / f"{base_name}_combined.png"
        cv2.imwrite(str(combined_mask_path), combined_mask)
        
        # Save overlay image
        overlay_path = overlay_dir / f"{base_name}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        # Save metadata
        metadata_path = metadata_dir / f"{base_name}_masks.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved masks to {masks_dir}")
        print(f"  Saved overlay to {overlay_path}")
        
        return metadata
    
    def process_single_bbox_file(self, bbox_file: str, image_path: str, output_dir: str):
        """Process a single YOLO bbox file and generate masks"""
        bbox_path = Path(bbox_file)
        base_name = bbox_path.stem
        
        print(f"\nProcessing {bbox_file}")
        
        # Generate masks
        masks_result = self.generate_masks_from_yolo_boxes(image_path, bbox_file)
        
        if masks_result:
            # Save masks
            metadata = self.save_masks(masks_result, output_dir, base_name)
            return metadata
        else:
            print(f"Failed to process {bbox_file}")
            return None
    
    def process_directory(self, bbox_dir: str, image_dir: str, output_dir: str, 
                         image_extension: str = ".jpg"):
        """
        Process all bbox files in a directory
        
        Args:
            bbox_dir: Directory containing YOLO format .txt files
            image_dir: Directory containing corresponding images
            output_dir: Output directory for masks
            image_extension: Extension of image files
        """
        bbox_path = Path(bbox_dir)
        image_path = Path(image_dir)
        
        # Find all txt files
        bbox_files = list(bbox_path.glob("*.txt"))
        
        print(f"Found {len(bbox_files)} bbox files to process")
        
        all_results = []
        
        for bbox_file in bbox_files:
            # Find corresponding image
            base_name = bbox_file.stem
            
            # Try different possible image locations
            possible_images = [
                image_path / f"{base_name}{image_extension}",
                image_path / f"{base_name}.png",
                image_path / f"{base_name}.jpeg",
            ]
            
            img_file = None
            for possible_img in possible_images:
                if possible_img.exists():
                    img_file = str(possible_img)
                    break
            
            if not img_file:
                print(f"Warning: No image found for {bbox_file}")
                continue
            
            result = self.process_single_bbox_file(str(bbox_file), img_file, output_dir)
            if result:
                all_results.append(result)
        
        # Save summary
        summary_path = Path(output_dir) / "segmentation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "total_files_processed": len(all_results),
                "results": all_results
            }, f, indent=2)
        
        print(f"\n✅ Completed! Processed {len(all_results)} files")
        print(f"Results saved to: {output_dir}")
        
        return all_results


def main():
    # Example usage
    
    # Define class names for your dataset (customize as needed)
    class_names = {
        0: "person",
        1: "bicycle", 
        2: "train",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "car",
        7: "truck",
        # Add more classes as needed
    }
    
    # Initialize SAM segmenter
    segmenter = SAMSegmentationFromBoxes(
        sam_checkpoint="sam_vit_h_4b8939.pth",  # Download from Facebook SAM repo
        model_type="vit_h",
        device="cuda" if torch.cuda.is_available() else "cpu",
        class_names=class_names
    )
    
    # Example 1: Process single bbox file
    bbox_file = "002320/new_folder/44_002320_000030.txt"
    image_file = "002320/images/000030.jpg"  # Adjust path as needed
    output_dir = "segmentation_output"
    
    segmenter.process_single_bbox_file(bbox_file, image_file, output_dir)
    
    # Example 2: Process entire directory
    # segmenter.process_directory(
    #     bbox_dir="002320/new_folder",  # Directory with .txt files
    #     image_dir="002320/images",      # Directory with corresponding images
    #     output_dir="segmentation_output",
    #     image_extension=".jpg"
    # )


if __name__ == "__main__":
    main()