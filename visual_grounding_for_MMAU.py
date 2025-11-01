import cv2
import numpy as np
import torch
import os
from PIL import Image
import json
import re
from tqdm import tqdm
from collections import defaultdict

from utils.grounding_utils.box_ops import np_box_iou

iou_thresholds = [0.3, 0.5]

def load_grounded_caption(json_path):
    """Load grounded caption JSON containing text and bounding boxes"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_boxes_from_json(grounded_data, frame_id):
    """
    Extract bounding boxes for a specific frame from grounded caption data
    Expected format: 
    {
        "caption": "...",
        "frames": {
            "frame_000001": [{"box": [x1, y1, x2, y2], "label": "..."}],
            ...
        }
    }
    """
    frame_key = f"frame_{frame_id:06d}"
    if "frames" in grounded_data and frame_key in grounded_data["frames"]:
        boxes = []
        for obj in grounded_data["frames"][frame_key]:
            if "box" in obj or "bbox" in obj:
                box = obj.get("box", obj.get("bbox"))
                boxes.append(box)
        print("boxes--------------:", boxes)
        return np.array(boxes) if boxes else np.array([])
    return np.array([])

def extract_boxes_from_mask(mask_path):
    """Extract bounding boxes from a binary mask image"""
    if not os.path.exists(mask_path):
        return np.array([])
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.array([])
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Convert to [x1, y1, x2, y2] format
        boxes.append([x, y, x + w, y + h])
    
    return np.array(boxes) if boxes else np.array([])

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    Boxes are in [x1, y1, x2, y2] format
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def merge_overlapping_boxes(boxes, iou_threshold=0.5):
    """
    Merge overlapping bounding boxes
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    merged_boxes = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
            
        current_box = boxes[i].copy()
        used[i] = True
        
        # Find all boxes that overlap with current box
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
                
            # Calculate IoU
            iou = calculate_iou(current_box, boxes[j])
            
            if iou > iou_threshold:
                # Merge boxes by taking the union
                current_box = [
                    min(current_box[0], boxes[j][0]),
                    min(current_box[1], boxes[j][1]),
                    max(current_box[2], boxes[j][2]),
                    max(current_box[3], boxes[j][3])
                ]
                used[j] = True
        
        merged_boxes.append(current_box)
    
    return merged_boxes

def parse_visualization_image(img_path):
    """
    Extract bounding boxes from visualization images with drawn bboxes
    This detects cyan/blue colored bounding boxes in the visualization
    """
    if not os.path.exists(img_path):
        return np.array([])
    
    img = cv2.imread(img_path)
    if img is None:
        return np.array([])
    
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for cyan/blue color (the bounding box color in your image)
    # Cyan typically has H: 85-95, S: 100-255, V: 100-255
    # Adjust these values based on the exact shade in your visualizations
    lower_cyan = np.array([85, 100, 100])
    upper_cyan = np.array([100, 255, 255])
    
    # Create mask for cyan color
    mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of the cyan-colored regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = 500  # Minimum area threshold to filter out noise
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out very small or very thin boxes (likely text or noise)
        if w < 20 or h < 20:
            continue
            
        # Convert to [x1, y1, x2, y2] format
        boxes.append([x, y, x + w, y + h])
    
    # Merge overlapping boxes (in case box edges are detected as separate contours)
    if len(boxes) > 1:
        boxes = merge_overlapping_boxes(boxes, iou_threshold=0.5)
    
    return np.array(boxes) if boxes else np.array([])

def determine_question_type(caption_text):
    """
    Determine if the caption is interrogative or declarative
    """
    interrogative_keywords = ['what', 'where', 'when', 'who', 'which', 'how', 'why', 'is', 'are', 'can', 'does', 'do', '?']
    
    caption_lower = caption_text.lower().strip()
    
    # Check for question mark
    if '?' in caption_text:
        return "interrogative"
    
    # Check if starts with interrogative word
    for keyword in interrogative_keywords:
        if caption_lower.startswith(keyword + ' '):
            return "interrogative"
    
    return "declarative"

def summarize_metrics(results):
    """Summarize metrics by question type"""
    categories = set(x["qtype"] for x in results.values())
    metrics = {}
    counter = {}
    
    for category in categories:
        metrics[category] = {"mIoU": 0}
        for thresh in iou_thresholds:
            metrics[category][f"mIoU@{thresh}"] = 0
        counter[category] = 0
    
    for x in results.values():
        qtype = x["qtype"]
        metrics[qtype]["mIoU"] += x["mIoU"]
        for thresh in iou_thresholds:
            metrics[qtype][f"mIoU@{thresh}"] += x[f"mIoU@{thresh}"]
        counter[qtype] += 1
    
    for category in categories:
        for key in metrics[category]:
            if counter[category] > 0:
                metrics[category][key] = metrics[category][key] / counter[category]
        print(f"\n{'='*50}")
        print(f"Category: {category}")
        print(f"{'='*50}")
        for key, value in metrics[category].items():
            print(f"{key}: {value:.4f}")
    
    return metrics

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Eval grounding from pre-generated outputs")
    
    parser.add_argument("--gt_base_dir", type=str, 
                       default="/data/users/apal/CAP-DATA_44-62/VideoGLaMM/VideoGLaMM/accident_analysis_output_val_ground_truth_with_visualization/temp",
                       help="Base directory containing ground truth outputs")
    parser.add_argument("--pred_base_dir", type=str,
                       default="/data/users/apal/CAP-DATA_44-62/VideoGLaMM/VideoGLaMM/accident_analysis_output_val_prediction_with_visualization/temp",
                       help="Base directory containing prediction outputs")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--box_extraction_method", type=str, default="json",
                       choices=["json", "mask", "visualization"],
                       help="Method to extract bounding boxes")
    
    return parser.parse_args()


if __name__=='__main__':
    
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of video IDs
    video_ids = sorted([d for d in os.listdir(args.gt_base_dir) 
                       if os.path.isdir(os.path.join(args.gt_base_dir, d))])
    
    print(f"Found {len(video_ids)} videos to evaluate")
    
    vid_metrics = {}
    
    # Loop through each video
    for video_id in tqdm(video_ids):
        
        try:
            gt_video_dir = os.path.join(args.gt_base_dir, video_id)
            pred_video_dir = os.path.join(args.pred_base_dir, video_id)
            
            # Check if prediction directory exists
            #print("gt_video_dir-------------:", gt_video_dir)
            #rint("pred_video_dir-------------:", pred_video_dir)
            if not os.path.exists(pred_video_dir):
                print(f"Warning: No prediction found for {video_id}, skipping...")
                continue
            
            # Load ground truth caption
            gt_caption_json = os.path.join(gt_video_dir, "grounded_captions", 
                                          f"{video_id}_grounded_caption.json")
            gt_caption_txt = os.path.join(gt_video_dir, "grounded_captions", 
                                         f"{video_id}_grounded_caption.txt")
            
            if os.path.exists(gt_caption_json):
                gt_data = load_grounded_caption(gt_caption_json)
                caption_text = gt_data.get("caption", "")
            elif os.path.exists(gt_caption_txt):
                with open(gt_caption_txt, 'r') as f:
                    caption_text = f.read().strip()
                gt_data = {"caption": caption_text}
            else:
                print(f"Warning: No caption found for {video_id}, skipping...")
                continue
            
            # Determine question type
            qtype = determine_question_type(caption_text)
            
            # Get list of frames
            gt_vis_dir = os.path.join(gt_video_dir, "visualizations")
            pred_vis_dir = os.path.join(pred_video_dir, "visualizations")
            
            gt_frames = sorted([f for f in os.listdir(gt_vis_dir) 
                              if f.startswith("frame_") and f.endswith(".jpg")])
            
            if not gt_frames:
                gt_frames = sorted([f for f in os.listdir(gt_vis_dir) 
                                  if f.startswith("frame_") and f.endswith(".png")])
            
            # Calculate IoU for each frame
            frame_ious = []
            frame_metrics = {}
            
            for frame_file in gt_frames:
                frame_num = int(re.search(r'frame_(\d+)', frame_file).group(1))
                
                # Extract ground truth boxes
                if args.box_extraction_method == "json":
                    gt_boxes = extract_boxes_from_json(gt_data, frame_num)
                elif args.box_extraction_method == "mask":
                    # Assuming masks are saved with specific naming
                    gt_mask_path = os.path.join(gt_vis_dir, f"mask_{frame_num:06d}.jpg")
                    gt_boxes = extract_boxes_from_mask(gt_mask_path)
                else:  # visualization
                    gt_img_path = os.path.join(gt_vis_dir, frame_file)
                    gt_boxes = parse_visualization_image(gt_img_path)
                
                # Extract prediction boxes
                pred_frame_path = os.path.join(pred_vis_dir, frame_file)
                if args.box_extraction_method == "json":
                    pred_caption_json = os.path.join(pred_video_dir, "grounded_captions",
                                                    f"{video_id}_grounded_caption.json")
                    if os.path.exists(pred_caption_json):
                        pred_data = load_grounded_caption(pred_caption_json)
                        pred_boxes = extract_boxes_from_json(pred_data, frame_num)
                    else:
                        pred_boxes = np.array([])
                elif args.box_extraction_method == "mask":
                    pred_mask_path = os.path.join(pred_vis_dir, f"mask_{frame_num:06d}.jpg")
                    pred_boxes = extract_boxes_from_mask(pred_mask_path)
                else:
                    pred_boxes = parse_visualization_image(pred_frame_path)
                

                #print("pred_boxes--------------:", pred_boxes)
                #print("gt_boxes--------------:", gt_boxes)
                # Calculate IoU
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    # Take the first box if multiple exist (or use Hungarian matching for multiple objects)
                    iou_matrix = np_box_iou(pred_boxes, gt_boxes)
                    iou = iou_matrix.max()  # Best matching IoU
                else:
                    iou = 0.0
                
                frame_ious.append(iou)
                frame_metrics[frame_num] = {
                    "iou": iou,
                    "gt_boxes": gt_boxes.tolist() if len(gt_boxes) > 0 else [],
                    "pred_boxes": pred_boxes.tolist() if len(pred_boxes) > 0 else []
                }
            
            # Calculate mean IoU for this video
            if frame_ious:
                mIoU = np.mean(frame_ious)
            else:
                mIoU = 0.0
            
            # Calculate recall at different thresholds
            recalls = {thresh: 0 for thresh in iou_thresholds}
            for thresh in iou_thresholds:
                if mIoU > thresh:
                    recalls[thresh] = 1
            
            # Store metrics
            curr_video_metrics = {
                "video_id": video_id,
                "caption": caption_text,
                "qtype": qtype,
                "mIoU": mIoU,
                "num_frames": len(frame_ious),
                "frame_metrics": frame_metrics,
            }
            
            curr_video_metrics.update({
                f"mIoU@{thresh}": recalls[thresh]
                for thresh in iou_thresholds
            })
            
            vid_metrics[video_id] = curr_video_metrics
            
            # Save individual video metrics
            video_output_file = os.path.join(args.output_dir, f"{video_id}_metrics.json")
            with open(video_output_file, 'w') as f:
                json.dump(curr_video_metrics, f, indent=2)
            
        except Exception as e:
            print(f"\033[91mError processing {video_id}: {e}\033[0m")
            import traceback
            traceback.print_exc()
            continue
    
    # Summarize metrics
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    summary_metrics = summarize_metrics(vid_metrics)
    
    # Save overall results
    output_file = os.path.join(args.output_dir, "overall_metrics.json")
    with open(output_file, 'w') as f:
        json.dump({
            "summary": summary_metrics,
            "per_video": vid_metrics
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")