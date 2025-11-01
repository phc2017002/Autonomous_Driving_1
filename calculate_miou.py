import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from tqdm import tqdm

def extract_bounding_boxes(image_path: str, debug: bool = False) -> Dict[str, List[int]]:
    """
    Extract bounding boxes from an image by detecting colored rectangles.
    
    Args:
        image_path: Path to the image file
        debug: If True, display intermediate results
    
    Returns:
        Dictionary of bounding boxes {label: [x1, y1, x2, y2]}
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for blue and cyan rectangles
    # Blue color range
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Cyan color range
    lower_cyan = np.array([80, 100, 100])
    upper_cyan = np.array([100, 255, 255])
    
    # Create masks for both colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask_blue, mask_cyan)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Combined Mask')
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and extract bounding boxes
    boxes = {}
    box_index = 1
    
    for contour in contours:
        # Filter by area to remove noise
        area = cv2.contourArea(contour)
        if area < 1000:  # Minimum area threshold
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio to filter out non-rectangular shapes
        aspect_ratio = w / h if h > 0 else 0
        
        # Store bounding box
        label = f"object_{box_index}"
        boxes[label] = [x, y, x + w, y + h]
        box_index += 1
    
    if debug:
        img_debug = img.copy()
        for label, box in boxes.items():
            cv2.rectangle(img_debug, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img_debug, label, (box[0], box[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
        plt.title('Detected Boxes')
        plt.tight_layout()
        plt.show()
    
    return boxes


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] coordinates of first box
        box2: [x1, y1, x2, y2] coordinates of second box
    
    Returns:
        IoU score (float between 0 and 1)
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Check if there's an intersection
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Compute union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def match_boxes(gt_boxes: Dict[str, List[int]], 
                pred_boxes: Dict[str, List[int]], 
                iou_threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """
    Match ground truth boxes with predicted boxes based on IoU.
    
    Args:
        gt_boxes: Dictionary of ground truth boxes {label: [x1, y1, x2, y2]}
        pred_boxes: Dictionary of predicted boxes {label: [x1, y1, x2, y2]}
        iou_threshold: Minimum IoU threshold for matching
    
    Returns:
        List of tuples (gt_label, pred_label, iou_score)
    """
    matches = []
    used_pred_boxes = set()
    
    for gt_label, gt_box in gt_boxes.items():
        best_iou = 0.0
        best_pred_label = None
        
        for pred_label, pred_box in pred_boxes.items():
            if pred_label in used_pred_boxes:
                continue
            
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_label = pred_label
        
        if best_iou >= iou_threshold and best_pred_label is not None:
            matches.append((gt_label, best_pred_label, best_iou))
            used_pred_boxes.add(best_pred_label)
    
    return matches


def compute_miou(gt_boxes: Dict[str, List[int]], 
                 pred_boxes: Dict[str, List[int]],
                 iou_threshold: float = 0.5) -> Dict:
    """
    Compute mean IoU (mIoU) between ground truth and predicted boxes.
    
    Args:
        gt_boxes: Dictionary of ground truth boxes {label: [x1, y1, x2, y2]}
        pred_boxes: Dictionary of predicted boxes {label: [x1, y1, x2, y2]}
        iou_threshold: Minimum IoU threshold for matching
    
    Returns:
        Dictionary with mIoU and individual IoU scores
    """
    matches = match_boxes(gt_boxes, pred_boxes, iou_threshold)
    
    if not matches:
        return {
            'miou': 0.0,
            'num_matches': 0,
            'num_gt_boxes': len(gt_boxes),
            'num_pred_boxes': len(pred_boxes),
            'matches': []
        }
    
    iou_scores = [match[2] for match in matches]
    miou = np.mean(iou_scores)
    
    return {
        'miou': miou,
        'num_matches': len(matches),
        'num_gt_boxes': len(gt_boxes),
        'num_pred_boxes': len(pred_boxes),
        'matches': [
            {'gt_label': m[0], 'pred_label': m[1], 'iou': m[2]} 
            for m in matches
        ]
    }


def process_frame_pair(gt_path: str, pred_path: str, frame_name: str, 
                      debug: bool = False) -> Dict:
    """
    Process a single pair of ground truth and predicted images.
    
    Args:
        gt_path: Path to ground truth image
        pred_path: Path to predicted image
        frame_name: Name of the frame for identification
        debug: If True, show debug visualizations
    
    Returns:
        Dictionary with results for this frame
    """
    try:
        # Extract bounding boxes from both images
        gt_boxes = extract_bounding_boxes(gt_path, debug=debug)
        pred_boxes = extract_bounding_boxes(pred_path, debug=debug)
        
        # Compute mIoU
        results = compute_miou(gt_boxes, pred_boxes, iou_threshold=0.5)
        
        # Add frame information
        results['frame'] = frame_name
        results['gt_path'] = gt_path
        results['pred_path'] = pred_path
        results['status'] = 'success'
        
        return results
        
    except Exception as e:
        return {
            'frame': frame_name,
            'gt_path': gt_path,
            'pred_path': pred_path,
            'status': 'error',
            'error': str(e),
            'miou': 0.0,
            'num_matches': 0,
            'num_gt_boxes': 0,
            'num_pred_boxes': 0,
            'matches': []
        }


def visualize_comparison_sample(gt_dir: str, pred_dir: str, frame_name: str, results: Dict):
    """
    Visualize a sample comparison between ground truth and predicted boxes.
    """
    gt_path = os.path.join(gt_dir, frame_name)
    pred_path = os.path.join(pred_dir, frame_name)
    
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        return
    
    gt_img = cv2.imread(gt_path)
    pred_img = cv2.imread(pred_path)
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Ground Truth - {frame_name}', fontsize=12)
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted - {frame_name}', fontsize=12)
    plt.axis('off')
    
    plt.suptitle(f'mIoU: {results["miou"]:.4f} | Matches: {results["num_matches"]}/{results["num_gt_boxes"]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def process_two_directories(gt_dir: str, 
                          pred_dir: str,
                          file_extension: str = 'jpg',
                          output_csv: str = None,
                          visualize_samples: int = 0) -> pd.DataFrame:
    """
    Process all matching frames between ground truth and prediction directories.
    
    Args:
        gt_dir: Path to directory containing ground truth frames
        pred_dir: Path to directory containing predicted frames
        file_extension: Extension of image files to process (default: 'jpg')
        output_csv: Path to save results CSV (optional)
        visualize_samples: Number of sample frames to visualize (0 = none)
    
    Returns:
        DataFrame with results for all frames
    """
    # Get all frame files from ground truth directory
    gt_pattern = os.path.join(gt_dir, f'*.{file_extension}')
    gt_files = sorted(glob.glob(gt_pattern))
    
    if not gt_files:
        print(f"No {file_extension} files found in ground truth directory: {gt_dir}")
        return pd.DataFrame()
    
    # Get all frame files from prediction directory
    pred_pattern = os.path.join(pred_dir, f'*.{file_extension}')
    pred_files = sorted(glob.glob(pred_pattern))
    
    if not pred_files:
        print(f"No {file_extension} files found in prediction directory: {pred_dir}")
        return pd.DataFrame()
    
    # Create mapping of filenames
    gt_files_dict = {os.path.basename(f): f for f in gt_files}
    pred_files_dict = {os.path.basename(f): f for f in pred_files}
    
    # Find matching frames
    gt_filenames = set(gt_files_dict.keys())
    pred_filenames = set(pred_files_dict.keys())
    matching_frames = sorted(gt_filenames.intersection(pred_filenames))
    
    print(f"\n📊 Frame Statistics:")
    print(f"   Ground Truth frames: {len(gt_files)}")
    print(f"   Predicted frames: {len(pred_files)}")
    print(f"   Matching frames: {len(matching_frames)}")
    
    if len(matching_frames) < len(gt_files):
        missing_in_pred = gt_filenames - pred_filenames
        if missing_in_pred:
            print(f"\n⚠️  Frames in GT but not in predictions: {len(missing_in_pred)}")
            if len(missing_in_pred) <= 5:
                for fname in sorted(missing_in_pred)[:5]:
                    print(f"     - {fname}")
    
    if len(matching_frames) < len(pred_files):
        extra_in_pred = pred_filenames - gt_filenames
        if extra_in_pred:
            print(f"\n⚠️  Frames in predictions but not in GT: {len(extra_in_pred)}")
            if len(extra_in_pred) <= 5:
                for fname in sorted(extra_in_pred)[:5]:
                    print(f"     - {fname}")
    
    # Process each matching frame pair
    all_results = []
    
    print(f"\n🔄 Processing {len(matching_frames)} matching frame pairs...")
    print("=" * 70)
    
    for i, frame_name in enumerate(tqdm(matching_frames, desc="Processing frames")):
        gt_path = gt_files_dict[frame_name]
        pred_path = pred_files_dict[frame_name]
        
        # Process this frame pair
        debug = i < visualize_samples
        results = process_frame_pair(gt_path, pred_path, frame_name, debug=debug)
        all_results.append(results)
        
        # Show sample visualization
        if i < visualize_samples:
            visualize_comparison_sample(gt_dir, pred_dir, frame_name, results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate overall statistics
    successful_frames = df[df['status'] == 'success']
    
    if not successful_frames.empty:
        overall_stats = {
            'total_frames': len(df),
            'successful_frames': len(successful_frames),
            'failed_frames': len(df) - len(successful_frames),
            'mean_miou': successful_frames['miou'].mean(),
            'std_miou': successful_frames['miou'].std(),
            'min_miou': successful_frames['miou'].min(),
            'max_miou': successful_frames['miou'].max(),
            'median_miou': successful_frames['miou'].median(),
            'total_gt_boxes': successful_frames['num_gt_boxes'].sum(),
            'total_pred_boxes': successful_frames['num_pred_boxes'].sum(),
            'total_matches': successful_frames['num_matches'].sum(),
            'avg_gt_boxes_per_frame': successful_frames['num_gt_boxes'].mean(),
            'avg_pred_boxes_per_frame': successful_frames['num_pred_boxes'].mean(),
            'avg_matches_per_frame': successful_frames['num_matches'].mean()
        }
        
        # Print results
        print("\n" + "=" * 70)
        print("📈 OVERALL RESULTS")
        print("=" * 70)
        print(f"Total frames processed: {overall_stats['total_frames']}")
        print(f"Successful: {overall_stats['successful_frames']}")
        print(f"Failed: {overall_stats['failed_frames']}")
        print("\n" + "-" * 70)
        print("📊 mIoU STATISTICS")
        print("-" * 70)
        print(f"Mean mIoU:   {overall_stats['mean_miou']:.4f}")
        print(f"Std mIoU:    {overall_stats['std_miou']:.4f}")
        print(f"Median mIoU: {overall_stats['median_miou']:.4f}")
        print(f"Min mIoU:    {overall_stats['min_miou']:.4f}")
        print(f"Max mIoU:    {overall_stats['max_miou']:.4f}")
        print("\n" + "-" * 70)
        print("📦 BOX STATISTICS")
        print("-" * 70)
        print(f"Total GT boxes:   {overall_stats['total_gt_boxes']}")
        print(f"Total Pred boxes: {overall_stats['total_pred_boxes']}")
        print(f"Total matches:    {overall_stats['total_matches']}")
        print(f"Avg GT boxes/frame:   {overall_stats['avg_gt_boxes_per_frame']:.2f}")
        print(f"Avg Pred boxes/frame: {overall_stats['avg_pred_boxes_per_frame']:.2f}")
        print(f"Avg matches/frame:    {overall_stats['avg_matches_per_frame']:.2f}")
        
        # Show per-frame results for first few and worst performing frames
        print("\n" + "-" * 70)
        print("📝 SAMPLE FRAME RESULTS (First 5)")
        print("-" * 70)
        for idx, row in successful_frames.head(5).iterrows():
            print(f"{row['frame']}: mIoU={row['miou']:.4f}, "
                  f"Matches={row['num_matches']}/{row['num_gt_boxes']}")
        
        print("\n" + "-" * 70)
        print("⚠️  WORST PERFORMING FRAMES (Bottom 5)")
        print("-" * 70)
        worst_frames = successful_frames.nsmallest(5, 'miou')
        for idx, row in worst_frames.iterrows():
            print(f"{row['frame']}: mIoU={row['miou']:.4f}, "
                  f"Matches={row['num_matches']}/{row['num_gt_boxes']}")
        
        print("\n" + "-" * 70)
        print("🏆 BEST PERFORMING FRAMES (Top 5)")
        print("-" * 70)
        best_frames = successful_frames.nlargest(5, 'miou')
        for idx, row in best_frames.iterrows():
            print(f"{row['frame']}: mIoU={row['miou']:.4f}, "
                  f"Matches={row['num_matches']}/{row['num_gt_boxes']}")
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
    
    # Create visualization of mIoU over frames
    if not successful_frames.empty:
        plt.figure(figsize=(15, 6))
        
        # Plot mIoU over frames
        plt.subplot(1, 2, 1)
        frames_numeric = range(len(successful_frames))
        plt.plot(frames_numeric, successful_frames['miou'].values, 'b-', alpha=0.7, linewidth=1.5)
        plt.axhline(y=overall_stats['mean_miou'], color='r', linestyle='--', 
                   label=f'Mean mIoU: {overall_stats["mean_miou"]:.4f}')
        plt.fill_between(frames_numeric, successful_frames['miou'].values, 
                        overall_stats['mean_miou'], 
                        where=(successful_frames['miou'].values >= overall_stats['mean_miou']),
                        color='green', alpha=0.3, interpolate=True)
        plt.fill_between(frames_numeric, successful_frames['miou'].values, 
                        overall_stats['mean_miou'], 
                        where=(successful_frames['miou'].values < overall_stats['mean_miou']),
                        color='red', alpha=0.3, interpolate=True)
        plt.xlabel('Frame Index')
        plt.ylabel('mIoU')
        plt.title('mIoU per Frame')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        # Histogram of mIoU values
        plt.subplot(1, 2, 2)
        n, bins, patches = plt.hist(successful_frames['miou'].values, bins=20, 
                                   edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=overall_stats['mean_miou'], color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {overall_stats["mean_miou"]:.4f}')
        plt.axvline(x=overall_stats['median_miou'], color='g', linestyle='--', 
                   linewidth=2, label=f'Median: {overall_stats["median_miou"]:.4f}')
        plt.xlabel('mIoU')
        plt.ylabel('Number of Frames')
        plt.title('Distribution of mIoU Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'mIoU Analysis - GT: {os.path.basename(gt_dir)} vs Pred: {os.path.basename(pred_dir)}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.dirname(output_csv) if output_csv else '.'
        plot_path = os.path.join(plot_dir, 'miou_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Analysis plot saved to: {plot_path}")
        plt.show()
    
    return df


# Main execution
if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <ground_truth_dir> <predictions_dir> [file_extension] [output_csv]")
        print("\nExample:")
        print("  python script.py accident_analysis_output_val_ground_truth/002065/visualizations accident_analysis_output_baseline_on_entire_video_val_data/002065/visualizations")
        print("  python script.py /path/to/gt_dir /path/to/pred_dir jpg results.csv")
        print("\nDefaults:")
        print("  file_extension: jpg")
        print("  output_csv: auto-generated based on directory names")
        sys.exit(1)
    
    # Parse arguments
    gt_directory = sys.argv[1]
    pred_directory = sys.argv[2]
    file_ext = sys.argv[3] if len(sys.argv) > 3 else 'jpg'
    output_csv = sys.argv[4] if len(sys.argv) > 4 else None
    
    # If no explicit output CSV, create one based on directory names
    if output_csv is None:
        gt_name = os.path.basename(os.path.dirname(os.path.normpath(gt_directory)))
        pred_name = os.path.basename(os.path.dirname(os.path.normpath(pred_directory)))
        output_csv = f'miou_results_{gt_name}_vs_{pred_name}.csv'
    
    print("=" * 70)
    print("🎯 BATCH mIoU COMPUTATION FOR VIDEO FRAMES")
    print("=" * 70)
    print(f"\n📁 Ground Truth Directory: {gt_directory}")
    print(f"📁 Predictions Directory: {pred_directory}")
    print(f"📄 File Extension: .{file_ext}")
    print(f"💾 Output CSV: {output_csv}")
    print("=" * 70)
    
    # Process all frames
    results_df = process_two_directories(
        gt_dir=gt_directory,
        pred_dir=pred_directory,
        file_extension=file_ext,
        output_csv=output_csv,
        visualize_samples=0  # Set to > 0 to visualize first N frame pairs
    )
    
    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("=" * 70)