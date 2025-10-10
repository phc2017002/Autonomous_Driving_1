import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import imageio.v3 as iio
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.spatial import ConvexHull

class AccidentObjectTracker:
    def __init__(self, 
                 api_key: str = None,
                 sam_checkpoint: str = "sam_vit_h_4b8939.pth",
                 sam_model_type: str = "vit_h",
                 device: str = "cuda"):
        """
        Initialize Accident Object Tracker with detection, segmentation, and tracking
        
        Args:
            api_key: API key for object detection
            sam_checkpoint: Path to SAM checkpoint
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        # Initialize SAM
        print("Loading SAM model...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # Initialize CoTracker
        print("Loading CoTracker3...")
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        
        # Initialize object detector (from your previous code)
        self.init_object_detector()
        
        # Define accident object colors for visualization
        self.accident_colors = {
            "accident_object_1": (255, 0, 0),      # Red
            "accident_object_2": (255, 127, 0),    # Orange
            "accident_object_3": (255, 255, 0),    # Yellow
            "accident_object_4": (255, 0, 255),    # Magenta
        }
        
        print("All models loaded successfully!")
    
    def init_object_detector(self):
        """Initialize the object detection API client"""
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        
        # Try to use OpenAI client
        self.use_openai_client = False
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self.use_openai_client = True
        except:
            self.use_openai_client = False
            
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string"""
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_local_image_url(self, image_path: Path) -> str:
        """Create a data URL for local image"""
        base64_image = self.encode_image_to_base64(image_path)
        mime_type = "image/jpeg" if image_path.suffix.lower() in ['', '.jpeg'] else "image/png"
        return f"data:{mime_type};base64,{base64_image}"
    
    def detect_accident_objects_initial_frames(self, 
                                              video_folder: str,
                                              accident_frame: int,
                                              num_detection_frames: int = 3) -> Dict[int, List[Dict]]:
        """
        Detect accident objects in the accident frame and subsequent frames
        
        Args:
            video_folder: Path to video frames folder
            accident_frame: Frame number where accident occurs
            num_detection_frames: Number of frames to use for detection (default 3)
            
        Returns:
            Dictionary mapping frame numbers to detected accident objects
        """
        detections_by_frame = {}
        
        # Process accident frame and next frames
        for offset in range(num_detection_frames):
            frame_num = accident_frame + offset
            frame_name = f"{frame_num:06d}"
            
            # Find image path
            image_path = Path(video_folder) / "images" / frame_name
            if not image_path.exists():
                image_path = Path(video_folder) / frame_name
            
            if not image_path.exists():
                print(f"Frame {frame_name} not found")
                continue
            
            print(f"\nDetecting accident objects in frame {frame_num} (offset {offset})...")
            
            # Create detection prompt
            prompt = self.create_accident_detection_prompt(frame_num, accident_frame)
            
            # Detect objects
            detected_objects = self.detect_objects_in_frame(image_path, prompt)
            
            # Filter for accident objects only
            accident_objects = [
                obj for obj in detected_objects 
                if obj.get("is_accident_related", False) or "accident_object" in obj.get("object_label", "")
            ]
            
            detections_by_frame[frame_num] = accident_objects
            print(f"  Found {len(accident_objects)} accident objects")
        
        return detections_by_frame
    
    def create_accident_detection_prompt(self, frame_number: int, accident_frame: int) -> str:
        """Create prompt for accident object detection"""
        prompt = f"""You are analyzing frame {frame_number} of a traffic accident that occurred at frame {accident_frame}.

Detect ONLY accident-related objects in this frame:
- Detached parts (tires, wheels, bumpers)
- Damaged/affected vehicles
- Debris on the road
- Vehicles taking evasive action

Return ONLY a JSON array with accident objects:
[
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "object_label": "accident_object_1",
    "type": "detached_tire",
    "description": "black tire rolling across lanes",
    "is_accident_related": true
  }}
]

Label objects as accident_object_1 through accident_object_4.
Return ONLY the JSON array."""
        
        return prompt
    
    def detect_objects_in_frame(self, image_path: Path, prompt: str) -> List[Dict]:
        """Detect objects in a single frame using the API"""
        try:
            image_url = self.create_local_image_url(image_path)
            message_content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt}
            ]
            
            if self.use_openai_client:
                completion = self.client.chat.completions.create(
                    model="qwen-vl-max",
                    messages=[{"role": "user", "content": message_content}],
                    max_tokens=2000,
                    temperature=0.1
                )
                response = completion.choices[0].message.content
            else:
                # Use requests fallback
                import requests
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "qwen-vl-max",
                    "messages": [{"role": "user", "content": message_content}],
                    "max_tokens": 2000,
                    "temperature": 0.1
                }
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                ).json()['choices'][0]['message']['content']
            
            # Parse response
            return self.parse_detection_response(response)
            
        except Exception as e:
            print(f"Error detecting objects: {e}")
            return []
    
    def parse_detection_response(self, response: str) -> List[Dict]:
        """Parse detection response to extract objects"""
        import re
        try:
            # Clean response
            clean_response = response
            if "```json" in clean_response:
                clean_response = re.sub(r'```json\s*', '', clean_response)
                clean_response = re.sub(r'```\s*', '', clean_response)
            
            clean_response = clean_response.strip()
            
            if clean_response.startswith('['):
                return json.loads(clean_response)
        except:
            pass
        
        return []
    
    def generate_tracking_points_from_detections(self,
                                                image: np.ndarray,
                                                detections: List[Dict],
                                                points_per_object: int = 10) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate tracking points from detected accident objects using SAM
        
        Args:
            image: Input image (H, W, 3)
            detections: List of detected objects with bboxes
            points_per_object: Number of tracking points per object
            
        Returns:
            tracking_points: Array of tracking points (N, 2)
            point_metadata: List of metadata for each point
        """
        # Set image for SAM
        self.sam_predictor.set_image(image)
        
        all_tracking_points = []
        point_metadata = []
        
        for det_idx, detection in enumerate(detections):
            bbox = detection.get("bbox_2d", [])
            if not bbox or len(bbox) != 4:
                continue
            
            # Generate mask using SAM
            input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            
            try:
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                
                mask = masks[0]  # Binary mask
                
                # Generate tracking points from mask
                # Method 1: Sample points uniformly from the mask
                mask_coords = np.where(mask > 0)
                num_mask_points = len(mask_coords[0])
                
                if num_mask_points > 0:
                    # Sample points from mask
                    num_points = min(points_per_object, num_mask_points)
                    indices = np.random.choice(num_mask_points, num_points, replace=False)
                    
                    # Get (x, y) coordinates
                    sampled_points = np.column_stack([
                        mask_coords[1][indices],  # x coordinates
                        mask_coords[0][indices]   # y coordinates
                    ])
                    
                    # Add points and metadata
                    for point in sampled_points:
                        all_tracking_points.append(point)
                        point_metadata.append({
                            "object_id": det_idx,
                            "object_label": detection.get("object_label", "unknown"),
                            "object_type": detection.get("type", "unknown"),
                            "description": detection.get("description", ""),
                            "original_bbox": bbox,
                            "unique_object_id": f"{detection.get('object_label', 'unknown')}_{det_idx}"
                        })
                
                # Method 2: Add corner and center points for better tracking
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                corner_points = [
                    [cx, cy],  # Center
                    [bbox[0], bbox[1]],  # Top-left
                    [bbox[2], bbox[3]],  # Bottom-right
                    [bbox[0], bbox[3]],  # Bottom-left
                    [bbox[2], bbox[1]],  # Top-right
                ]
                
                for point in corner_points[:min(5, points_per_object // 2)]:
                    all_tracking_points.append(point)
                    point_metadata.append({
                        "object_id": det_idx,
                        "object_label": detection.get("object_label", "unknown"),
                        "object_type": detection.get("type", "unknown"),
                        "description": detection.get("description", ""),
                        "original_bbox": bbox,
                        "is_corner": True,
                        "unique_object_id": f"{detection.get('object_label', 'unknown')}_{det_idx}"
                    })
                    
            except Exception as e:
                print(f"Error generating mask for object {det_idx}: {e}")
                continue
        
        if all_tracking_points:
            tracking_points = np.array(all_tracking_points, dtype=np.float32)
        else:
            tracking_points = np.array([], dtype=np.float32).reshape(0, 2)
        
        return tracking_points, point_metadata
    
    def compute_bounding_boxes_from_tracks(self,
                                          tracks: np.ndarray,
                                          visibility: np.ndarray,
                                          point_metadata: List[Dict],
                                          method: str = "minmax") -> Dict[int, Dict[str, np.ndarray]]:
        """
        Compute bounding boxes from tracked points
        
        Args:
            tracks: Tracked points array (T, N, 2)
            visibility: Visibility array (T, N)
            point_metadata: Metadata for each point
            method: Method to compute bbox ("minmax", "convex_hull", or "expanded")
            
        Returns:
            Dictionary mapping frame index to object bounding boxes
        """
        T, N, _ = tracks.shape
        
        # Group points by unique object ID
        object_points = {}
        for idx, meta in enumerate(point_metadata):
            obj_id = meta.get("unique_object_id", f"unknown_{idx}")
            if obj_id not in object_points:
                object_points[obj_id] = {
                    "indices": [],
                    "metadata": meta
                }
            object_points[obj_id]["indices"].append(idx)
        
        # Compute bounding boxes for each frame
        frame_bboxes = {}
        
        for t in range(T):
            bboxes = {}
            
            for obj_id, obj_data in object_points.items():
                # Get visible points for this object at this frame
                visible_points = []
                for idx in obj_data["indices"]:
                    if visibility[t, idx] > 0.5:
                        visible_points.append(tracks[t, idx])
                
                if len(visible_points) >= 2:  # Need at least 2 points for a bbox
                    visible_points = np.array(visible_points)
                    
                    if method == "minmax":
                        # Simple min-max bounding box
                        x_min, y_min = visible_points.min(axis=0)
                        x_max, y_max = visible_points.max(axis=0)
                        
                    elif method == "convex_hull":
                        # Convex hull then bounding box
                        try:
                            if len(visible_points) >= 3:
                                hull = ConvexHull(visible_points)
                                hull_points = visible_points[hull.vertices]
                                x_min, y_min = hull_points.min(axis=0)
                                x_max, y_max = hull_points.max(axis=0)
                            else:
                                x_min, y_min = visible_points.min(axis=0)
                                x_max, y_max = visible_points.max(axis=0)
                        except:
                            x_min, y_min = visible_points.min(axis=0)
                            x_max, y_max = visible_points.max(axis=0)
                    
                    elif method == "expanded":
                        # Expand bbox by a margin based on original size
                        x_min, y_min = visible_points.min(axis=0)
                        x_max, y_max = visible_points.max(axis=0)
                        
                        # Expand by 20% on each side
                        width = x_max - x_min
                        height = y_max - y_min
                        margin_x = width * 0.2
                        margin_y = height * 0.2
                        
                        x_min -= margin_x
                        x_max += margin_x
                        y_min -= margin_y
                        y_max += margin_y
                    
                    else:
                        x_min, y_min = visible_points.min(axis=0)
                        x_max, y_max = visible_points.max(axis=0)
                    
                    # Store bbox with metadata
                    bboxes[obj_id] = {
                        "bbox": np.array([x_min, y_min, x_max, y_max]),
                        "num_points": len(visible_points),
                        "center": visible_points.mean(axis=0),
                        "object_label": obj_data["metadata"].get("object_label"),
                        "object_type": obj_data["metadata"].get("object_type"),
                        "description": obj_data["metadata"].get("description")
                    }
            
            frame_bboxes[t] = bboxes
        
        return frame_bboxes
    
    def track_accident_objects(self,
                              video_folder: str,
                              initial_detections: Dict[int, List[Dict]],
                              start_frame: int,
                              end_frame: int,
                              output_dir: str = "tracking_output") -> Dict:
        """
        Track accident objects through video using CoTracker
        
        Args:
            video_folder: Path to video frames
            initial_detections: Initial detections by frame number
            start_frame: First frame to track
            end_frame: Last frame to track
            output_dir: Output directory for results
            
        Returns:
            Tracking results dictionary with bounding boxes
        """
        print(f"\n{'='*60}")
        print(f"Starting CoTracker tracking from frame {start_frame} to {end_frame}")
        print(f"{'='*60}")
        
        # Load video frames
        frames_list = []
        frame_paths = []
        
        for frame_num in range(start_frame, end_frame + 1):
            frame_name = f"{frame_num:06d}"
            frame_path = Path(video_folder) / "images" / frame_name
            if not frame_path.exists():
                frame_path = Path(video_folder) / frame_name
            
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
                frame_paths.append(str(frame_path))
            else:
                print(f"Warning: Frame {frame_name} not found")
        
        if len(frames_list) < 2:
            print("Not enough frames for tracking")
            return None
        
        # Convert frames to tensor
        video_tensor = torch.tensor(np.stack(frames_list)).permute(0, 3, 1, 2).float()
        video_tensor = video_tensor.unsqueeze(0).to(self.device)  # B T C H W
        
        # Collect all initial tracking points from detection frames
        all_initial_points = []
        all_point_metadata = []
        
        # Track unique object IDs across frames
        object_id_counter = 0
        object_id_mapping = {}
        
        for frame_num in sorted(initial_detections.keys()):
            if frame_num < start_frame or frame_num > end_frame:
                continue
            
            frame_idx = frame_num - start_frame
            if frame_idx >= len(frames_list):
                continue
            
            detections = initial_detections[frame_num]
            if detections:
                # Assign unique IDs to objects
                for det in detections:
                    obj_label = det.get("object_label", "unknown")
                    if obj_label not in object_id_mapping:
                        object_id_mapping[obj_label] = object_id_counter
                        object_id_counter += 1
                
                # Generate tracking points from detections
                points, metadata = self.generate_tracking_points_from_detections(
                    frames_list[frame_idx],
                    detections,
                    points_per_object=8
                )
                
                if len(points) > 0:
                    # Add time dimension to points
                    points_with_time = np.column_stack([
                        np.full(len(points), frame_idx),  # Time index
                        points  # x, y coordinates
                    ])
                    
                    all_initial_points.append(points_with_time)
                    all_point_metadata.extend(metadata)
        
        if not all_initial_points:
            print("No tracking points generated from detections")
            return None
        
        # Combine all initial points
        queries = np.vstack(all_initial_points)
        queries_tensor = torch.tensor(queries).float().to(self.device)  # N, 3 (t, x, y)
        
        print(f"Tracking {len(queries)} points across {len(frames_list)} frames...")
        
        # Run CoTracker with queries
        pred_tracks, pred_visibility = self.cotracker(
            video_tensor,
            queries=queries_tensor[None]  # Add batch dimension
        )
        
        # Convert results back to numpy
        pred_tracks = pred_tracks[0].cpu().numpy()  # Remove batch dimension: T N 2
        pred_visibility = pred_visibility[0].cpu().numpy()  # T N or T N 1
        
        # Fix visibility shape if needed
        if len(pred_visibility.shape) == 3:
            pred_visibility = pred_visibility.squeeze(-1)  # Remove last dimension if T N 1
        
        print(f"Tracks shape: {pred_tracks.shape}")
        print(f"Visibility shape: {pred_visibility.shape}")
        
        # Compute bounding boxes from tracks
        print("Computing bounding boxes from tracked points...")
        frame_bboxes = self.compute_bounding_boxes_from_tracks(
            pred_tracks,
            pred_visibility,
            all_point_metadata,
            method="expanded"  # Use expanded method for better coverage
        )
        
        # Organize tracking results
        tracking_results = {
            "video_folder": video_folder,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": len(frames_list),
            "num_tracked_points": len(queries),
            "tracks": pred_tracks,
            "visibility": pred_visibility,
            "point_metadata": all_point_metadata,
            "frame_paths": frame_paths,
            "initial_detections": initial_detections,
            "frame_bboxes": frame_bboxes,  # Add bounding boxes
            "object_id_mapping": object_id_mapping
        }
        
        # Save tracking results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw tracking data
        np.save(output_path / "tracks.npy", pred_tracks)
        np.save(output_path / "visibility.npy", pred_visibility)
        
        # Save bounding boxes for each frame
        bbox_data = {}
        for t, bboxes in frame_bboxes.items():
            frame_num = start_frame + t
            bbox_data[frame_num] = {}
            for obj_id, bbox_info in bboxes.items():
                bbox_data[frame_num][obj_id] = {
                    "bbox": bbox_info["bbox"].tolist(),
                    "center": bbox_info["center"].tolist(),
                    "num_points": bbox_info["num_points"],
                    "object_label": bbox_info["object_label"],
                    "object_type": bbox_info["object_type"]
                }
        
        with open(output_path / "bounding_boxes.json", 'w') as f:
            json.dump(bbox_data, f, indent=2)
        
        # Save metadata
        with open(output_path / "tracking_metadata.json", 'w') as f:
            json.dump({
                "video_folder": video_folder,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": len(frames_list),
                "num_tracked_points": len(queries),
                "point_metadata": all_point_metadata,
                "frame_paths": frame_paths,
                "object_id_mapping": object_id_mapping
            }, f, indent=2)
        
        print(f"Tracking complete! Results saved to {output_path}")
        print(f"  - Bounding boxes saved to: {output_path}/bounding_boxes.json")
        
        return tracking_results
    
    def visualize_tracking_with_bboxes(self,
                                      tracking_results: Dict,
                                      output_dir: str = "tracking_visualization",
                                      save_video: bool = True,
                                      show_points: bool = True,
                                      show_trails: bool = True,
                                      fps: int = 10):
        """
        Visualize tracking results with bounding boxes and optional points/trails
        
        Args:
            tracking_results: Tracking results from track_accident_objects
            output_dir: Output directory for visualizations
            save_video: Whether to save as video
            show_points: Whether to show individual tracking points
            show_trails: Whether to show point trails
            fps: Frames per second for video
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tracks = tracking_results["tracks"]  # T N 2
        visibility = tracking_results["visibility"]  # T N
        point_metadata = tracking_results["point_metadata"]
        frame_paths = tracking_results["frame_paths"]
        frame_bboxes = tracking_results["frame_bboxes"]
        
        # Handle visibility shape
        if len(visibility.shape) == 3:
            visibility = visibility.squeeze(-1)
        
        frames_with_tracks = []
        
        print("Visualizing tracking results with bounding boxes...")
        
        for t in tqdm(range(len(frame_paths)), desc="Creating visualizations"):
            # Load frame
            frame = cv2.imread(frame_paths[t])
            if frame is None:
                print(f"Warning: Could not load frame {frame_paths[t]}")
                continue
            
            # Create overlay image
            overlay = frame.copy()
            
            # Draw bounding boxes for this frame
            if t in frame_bboxes:
                for obj_id, bbox_info in frame_bboxes[t].items():
                    bbox = bbox_info["bbox"]
                    obj_label = bbox_info.get("object_label", "unknown")
                    obj_type = bbox_info.get("object_type", "unknown")
                    
                    # Get color for this object
                    if obj_label in self.accident_colors:
                        color = self.accident_colors[obj_label]
                    else:
                        color = (0, 255, 0)  # Default green
                    
                    # Draw bounding box
                    cv2.rectangle(overlay,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                color, 3)
                    
                    # Add label
                    label = f"{obj_label}: {obj_type}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw label background
                    cv2.rectangle(overlay,
                                (int(bbox[0]), int(bbox[1] - label_size[1] - 10)),
                                (int(bbox[0] + label_size[0]), int(bbox[1])),
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(overlay, label,
                              (int(bbox[0]), int(bbox[1] - 5)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw center point
                    center = bbox_info["center"]
                    cv2.circle(overlay, (int(center[0]), int(center[1])), 5, color, -1)
            
            # Optionally draw individual tracking points
            if show_points:
                for n in range(tracks.shape[1]):
                    if visibility[t, n] > 0.5:
                        point = tracks[t, n]
                        meta = point_metadata[n] if n < len(point_metadata) else {}
                        obj_label = meta.get("object_label", "unknown")
                        
                        # Get color
                        if obj_label in self.accident_colors:
                            color = self.accident_colors[obj_label]
                        else:
                            color = (0, 255, 0)
                        
                        # Draw point
                        cv2.circle(overlay, (int(point[0]), int(point[1])), 3, color, -1)
                        
                        # Draw trail if enabled
                        if show_trails and t > 0:
                            trail_points = []
                            for prev_t in range(max(0, t - 10), t):
                                if visibility[prev_t, n] > 0.5:
                                    trail_points.append(tracks[prev_t, n])
                            
                            if len(trail_points) > 1:
                                trail_points = np.array(trail_points, dtype=np.int32)
                                cv2.polylines(overlay, [trail_points], False, color, 2)
            
            # Add frame info
            frame_num = tracking_results["start_frame"] + t
            cv2.putText(overlay, f"Frame {frame_num}",
                      (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add statistics
            num_objects = len(frame_bboxes.get(t, {}))
            cv2.putText(overlay, f"Tracked Objects: {num_objects}",
                      (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save frame
            frame_path = output_path / f"frame_{frame_num:06d}"
            cv2.imwrite(str(frame_path), overlay)
            
            # Collect for video
            if save_video:
                frames_with_tracks.append(overlay)
        
        # Create video if requested
        if save_video and frames_with_tracks:
            video_path = output_path / "tracking_video_with_bboxes.mp4"
            height, width = frames_with_tracks[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for frame in frames_with_tracks:
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Video saved to {video_path}")
        
        print(f"Visualizations saved to {output_path}")
    
    def process_accident_video_complete(self,
                                       video_folder: str,
                                       accident_frame: int,
                                       total_frames: int,
                                       output_base_dir: str = "accident_tracking_output"):
        """
        Complete pipeline: Detect accident objects, track them, and compute bounding boxes
        
        Args:
            video_folder: Path to video frames
            accident_frame: Frame where accident occurs
            total_frames: Total number of frames in video
            output_base_dir: Base output directory
        """
        video_name = Path(video_folder).name
        output_dir = Path(output_base_dir) / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing accident video: {video_name}")
        print(f"Accident frame: {accident_frame}")
        print(f"Total frames: {total_frames}")
        print(f"{'='*60}")
        
        # Step 1: Detect accident objects in initial frames
        print("\n📍 Step 1: Detecting accident objects in initial frames...")
        initial_detections = self.detect_accident_objects_initial_frames(
            video_folder,
            accident_frame,
            num_detection_frames=3  # Use frames 27, 28, 29
        )
        
        # Save initial detections
        with open(output_dir / "initial_detections.json", 'w') as f:
            detections_for_json = {
                str(k): v for k, v in initial_detections.items()
            }
            json.dump(detections_for_json, f, indent=2)
        
        # Check if we have detections
        total_detected = sum(len(dets) for dets in initial_detections.values())
        if total_detected == 0:
            print("❌ No accident objects detected. Exiting.")
            return None
        
        print(f"✅ Detected {total_detected} accident objects across initial frames")
        
        # Step 2: Track accident objects and compute bounding boxes
        print("\n📍 Step 2: Tracking accident objects with CoTracker and computing bounding boxes...")
        tracking_results = self.track_accident_objects(
            video_folder,
            initial_detections,
            start_frame=accident_frame,
            end_frame=min(accident_frame + 50, total_frames),  # Track for 50 frames or until end
            output_dir=str(output_dir / "tracking_data")
        )
        
        if not tracking_results:
            print("❌ Tracking failed")
            return None
        
        print("✅ Tracking complete with bounding boxes")
        
        # Step 3: Visualize tracking results with bounding boxes
        print("\n📍 Step 3: Creating visualizations with bounding boxes...")
        self.visualize_tracking_with_bboxes(
            tracking_results,
            output_dir=str(output_dir / "visualizations"),
            save_video=True,
            show_points=False,  # Set to True to also show tracking points
            show_trails=False,  # Set to True to show point trails
            fps=10
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Processing complete for {video_name}")
        print(f"📁 Results saved to: {output_dir}")
        print(f"  - initial_detections.json: Detected accident objects")
        print(f"  - tracking_data/bounding_boxes.json: Frame-by-frame bounding boxes")
        print(f"  - tracking_data/tracks.npy: Raw tracking points")
        print(f"  - visualizations/: Tracking visualizations with bounding boxes")
        print(f"{'='*60}")
        
        return {
            "video_name": video_name,
            "initial_detections": initial_detections,
            "tracking_results": tracking_results,
            "output_dir": str(output_dir)
        }


def main():
    # Configuration
    BASE_DIR = "."  # Adjust to your directory
    
    # Initialize the accident object tracker
    tracker = AccidentObjectTracker(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # Your API key
        sam_checkpoint=f"{BASE_DIR}/sam_vit_h_4b8939.pth",  # Path to SAM checkpoint
        sam_model_type="vit_h",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Process video 002320
    video_folder = f"{BASE_DIR}/000909"  # Path to video frames
    accident_frame = 27
    total_frames = 113
    
    # Run complete pipeline
    results = tracker.process_accident_video_complete(
        video_folder=video_folder,
        accident_frame=accident_frame,
        total_frames=total_frames,
        output_base_dir="accident_tracking_output"
    )
    
    if results:
        print("\n🎉 Success! Check the output directory for:")
        print("  1. Initial accident object detections")
        print("  2. Frame-by-frame bounding boxes (JSON format)")
        print("  3. Visualization video with bounding boxes")
        print("\nBounding boxes are computed from tracked points and saved as:")
        print("  - JSON file with bbox coordinates for each frame")
        print("  - Visual overlays showing the tracked bounding boxes")


if __name__ == "__main__":
    main()