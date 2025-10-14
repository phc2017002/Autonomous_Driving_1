import cv2
import numpy as np
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Tuple, Optional, Union, Any
import re
import os
from pathlib import Path
from dataclasses import dataclass
from glob import glob
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import imageio.v3 as iio
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

@dataclass
class VideoMetadata:
    """Data class for video metadata"""
    video_id: str
    weather: int  # 1=sunny, 2=rainy, 3=snowy, 4=foggy
    light: int    # 1=day, 2=night
    scene: int    # 1=highway, 2=tunnel, 3=mountain, 4=urban, 5=rural
    road: int     # 1=arterials, 2=curve, 3=intersection, 4=T-junction, 5=ramp
    accident: bool
    abnormal_start: int
    abnormal_end: int
    accident_frame: int
    total_frames: int
    dense_caption: str
    causes: str
    measures: str
    
    def get_weather_str(self) -> str:
        return {1: "sunny", 2: "rainy", 3: "snowy", 4: "foggy"}.get(self.weather, "unknown")
    
    def get_light_str(self) -> str:
        return {1: "day", 2: "night"}.get(self.light, "unknown")
    
    def get_scene_str(self) -> str:
        return {1: "highway", 2: "tunnel", 3: "mountain", 4: "urban", 5: "rural"}.get(self.scene, "unknown")
    
    def get_road_str(self) -> str:
        return {1: "arterials", 2: "curve", 3: "intersection", 4: "T-junction", 5: "ramp"}.get(self.road, "unknown")


class AccidentVideoAnalyzer:
    """Combined system for grounded caption generation and accident object tracking"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 sam_checkpoint: Optional[str] = None,
                 sam_model_type: str = "vit_h",
                 device: str = "cuda",
                 enable_tracking: bool = True):
        """
        Initialize the combined analyzer
        
        Args:
            api_key: DASHSCOPE API key
            sam_checkpoint: Path to SAM checkpoint (for tracking)
            sam_model_type: SAM model type
            device: Device for models
            enable_tracking: Whether to enable tracking functionality
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not provided and not found in environment variables")
        
        self.device = device
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.enable_tracking = enable_tracking
        
        # Initialize API client
        self.use_openai_client = False
        try:
            from openai import OpenAI
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self.use_openai_client = True
                print("Using OpenAI client for DASHSCOPE API")
            except:
                self.use_openai_client = False
        except ImportError:
            print("OpenAI library not found, using requests instead")
            self.use_openai_client = False
        
        if not self.use_openai_client:
            print("Will use requests library for DASHSCOPE API calls")
        
        # Initialize tracking models if enabled
        if enable_tracking and sam_checkpoint:
            self.init_tracking_models(sam_checkpoint, sam_model_type)
        else:
            self.sam_predictor = None
            self.cotracker = None
            if enable_tracking:
                print("Warning: Tracking enabled but SAM checkpoint not provided")
        
        # Define accident object colors for visualization
        self.accident_colors = {
            "accident_object_1": (255, 0, 0),      # Red
            "accident_object_2": (255, 127, 0),    # Orange
            "accident_object_3": (255, 255, 0),    # Yellow
            "accident_object_4": (255, 0, 255),    # Magenta
        }
        
        # Embedded metadata for videos
        self.embedded_metadata = {
            "000909": {
                "Video": "000909",
                "weather(sunny,rainy,snowy,foggy)1-4": 1,
                "light(day,night)1-2": 2,
                "scenes(highway,tunnel,mountain,urban,rural)1-5": 4,
                "linear(arterials,curve,intersection,T-junction,ramp) 1-5": 1,
                "type": 51,
                "whether an accident occurred (1/0)": 1,
                "abnormal start frame": 19,
                "abnormal end frame": 44,
                "accident frame": 34,
                "total frames": 63,
                "[0,tai]": 19,
                "[tai,tco]": 15,
                "[tai,tae]": 25,
                "[tco,tae]": 10,
                "[tae,end]": 19,
                "texts": "[CLS]a vehicle makes an evasive action[SEP]",
                "causes": "The vehicle does not notice the motorcycles when open the car door",
                "measures": "The driver shall remind the passengers in advance to see the traffic of pedestrians and vehicles in front of and behind the vehicle, and can only open the door and get off after confirming safety"
            }
        }
    
    def init_tracking_models(self, sam_checkpoint: str, sam_model_type: str):
        """Initialize SAM and CoTracker models for tracking"""
        try:
            # Initialize SAM
            print("Loading SAM model...")
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            
            # Initialize CoTracker
            print("Loading CoTracker3...")
            self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
            
            print("Tracking models loaded successfully!")
        except Exception as e:
            print(f"Error loading tracking models: {e}")
            self.sam_predictor = None
            self.cotracker = None
    
    def load_video_metadata(self, video_folder: Path) -> VideoMetadata:
        """Load metadata from embedded data or video folder"""
        video_id = video_folder.name
        
        # Check if we have embedded metadata for this video
        if video_id in self.embedded_metadata:
            print(f"Using embedded metadata for video {video_id}")
            data = self.embedded_metadata[video_id]
            
            # Create VideoMetadata from embedded data
            metadata = VideoMetadata(
                video_id=data["Video"],
                weather=data["weather(sunny,rainy,snowy,foggy)1-4"],
                light=data["light(day,night)1-2"],
                scene=data["scenes(highway,tunnel,mountain,urban,rural)1-5"],
                road=data["linear(arterials,curve,intersection,T-junction,ramp) 1-5"],
                accident=bool(data["whether an accident occurred (1/0)"]),
                abnormal_start=data["abnormal start frame"],
                abnormal_end=data["abnormal end frame"],
                accident_frame=data["accident frame"],
                total_frames=data["total frames"],
                dense_caption=data.get("texts", "[CLS]a vehicle makes an evasive action[SEP]"),
                causes=data["causes"],
                measures=data["measures"]
            )
            return metadata
        
        # Otherwise, try to load from metadata.json file
        metadata_file = video_folder / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = VideoMetadata(
                video_id=data.get("video_id", video_folder.name),
                weather=data.get("weather", 1),
                light=data.get("light", 1),
                scene=data.get("scene", 1),
                road=data.get("road", 1),
                accident=data.get("accident", False),
                abnormal_start=data.get("abnormal_start", 0),
                abnormal_end=data.get("abnormal_end", 0),
                accident_frame=data.get("accident_frame", 0),
                total_frames=data.get("total_frames", 0),
                dense_caption=data.get("dense_caption", ""),
                causes=data.get("causes", ""),
                measures=data.get("measures", "")
            )
        else:
            print(f"Warning: No metadata found for {video_folder}")
            frames = self.get_available_frames(video_folder)
            metadata = VideoMetadata(
                video_id=video_folder.name,
                weather=1,
                light=1,
                scene=4,
                road=1,
                accident=False,
                abnormal_start=0,
                abnormal_end=0,
                accident_frame=0,
                total_frames=len(frames),
                dense_caption="Traffic video footage",
                causes="",
                measures=""
            )
        
        return metadata
    
    def get_available_frames(self, video_folder: Path) -> List[Path]:
        """Get all available frame files from video folder"""
        frames = []
        
        # Check if there's an 'images' subdirectory
        images_dir = video_folder / "images"
        if images_dir.exists():
            search_dir = images_dir
        else:
            search_dir = video_folder
        
        # Try different common frame naming patterns
        patterns = [
            "*.jpg", "*.jpeg", "*.png",
            "frame_*.jpg", "frame_*.png",
            "*_frame.jpg", "*_frame.png",
            "0*.jpg", "0*.png"
        ]
        
        for pattern in patterns:
            found_frames = list(search_dir.glob(pattern))
            if found_frames:
                frames.extend(found_frames)
        
        # Remove duplicates and sort
        frames = list(set(frames))
        
        # Sort frames numerically
        def extract_frame_number(path):
            numbers = re.findall(r'\d+', path.stem)
            if numbers:
                return int(numbers[0])
            return 0
        
        frames.sort(key=extract_frame_number)
        
        return frames
    
    def get_all_frame_paths(self, video_folder: Path) -> List[Tuple[int, Path]]:
        """Get ALL frames with their frame numbers for tracking/visualization"""
        available_frames = self.get_available_frames(video_folder)
        
        frame_paths_with_numbers = []
        for frame_path in available_frames:
            # Extract frame number from filename
            numbers = re.findall(r'\d+', frame_path.stem)
            if numbers:
                frame_num = int(numbers[0])
                frame_paths_with_numbers.append((frame_num, frame_path))
        
        return sorted(frame_paths_with_numbers, key=lambda x: x[0])
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_local_image_url(self, image_path: str) -> str:
        """Create a data URL for local image"""
        path = Path(image_path)
        base64_image = self.encode_image_to_base64(path)
        mime_type = "image/jpeg" if path.suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
        return f"data:{mime_type};base64,{base64_image}"
    
    def call_api_with_requests_multi_image(self, images_with_prompts: List[Dict], 
                                          model="qwen-vl-max", max_tokens=4000, temperature=0.1):
        """Call DASHSCOPE API with multiple images"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        message_content = []
        for item in images_with_prompts:
            if "image_path" in item:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": self.create_local_image_url(item["image_path"])}
                })
            if "text" in item:
                message_content.append({
                    "type": "text",
                    "text": item["text"]
                })
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")
    
    # ========== GROUNDED CAPTION GENERATION METHODS ==========
    
    def select_representative_frames(self, video_metadata: VideoMetadata, 
                                    available_frames: List[Path], 
                                    max_frames: int = 15) -> List[Tuple[int, Path]]:
        """Select representative frames for caption generation (NOT for video composition)"""
        total_frames = len(available_frames)
        
        if video_metadata.accident and video_metadata.abnormal_start > 0:
            # Key moments for accident videos
            key_moments = [
                1,
                max(1, video_metadata.abnormal_start - 2),
                video_metadata.abnormal_start,
                (video_metadata.abnormal_start + video_metadata.accident_frame) // 2,
                video_metadata.accident_frame - 1,
                video_metadata.accident_frame,
                video_metadata.accident_frame + 1,
                (video_metadata.accident_frame + video_metadata.abnormal_end) // 2,
                video_metadata.abnormal_end,
                min(video_metadata.total_frames, video_metadata.abnormal_end + 5),
            ]
            
            selected_frames = []
            for frame_num in key_moments:
                for frame_path in available_frames:
                    numbers = re.findall(r'\d+', frame_path.stem)
                    if numbers and int(numbers[0]) == frame_num:
                        selected_frames.append((frame_num, frame_path))
                        break
            
            if len(selected_frames) < max_frames // 2:
                step = max(1, total_frames // max_frames)
                selected_frames = []
                for i in range(0, total_frames, step):
                    if i < total_frames:
                        frame_path = available_frames[i]
                        numbers = re.findall(r'\d+', frame_path.stem)
                        frame_num = int(numbers[0]) if numbers else i + 1
                        selected_frames.append((frame_num, frame_path))
                    if len(selected_frames) >= max_frames:
                        break
        else:
            step = max(1, total_frames // max_frames)
            selected_frames = []
            for i in range(0, total_frames, step):
                if i < total_frames:
                    frame_path = available_frames[i]
                    numbers = re.findall(r'\d+', frame_path.stem)
                    frame_num = int(numbers[0]) if numbers else i + 1
                    selected_frames.append((frame_num, frame_path))
                if len(selected_frames) >= max_frames:
                    break
        
        return sorted(selected_frames, key=lambda x: x[0])[:max_frames]
    
    def create_comprehensive_grounding_prompt(self, 
                                             video_metadata: VideoMetadata,
                                             num_frames: int) -> str:
        """Create prompt for two-section grounded caption"""
        
        if video_metadata.video_id == "000909":
            incident_description = """This is a door-opening accident where a parked vehicle opens its door into the path of passing motorcycles.
The incident occurs at night on an urban arterial road."""
        else:
            incident_description = video_metadata.causes if video_metadata.accident else "No incident"
        
        prompt = f"""Generate a comprehensive grounded caption in TWO DISTINCT SECTIONS:

VIDEO INFO:
- Accident Frame: {video_metadata.accident_frame}
- Total Frames: {video_metadata.total_frames}
- Incident: {incident_description}

SECTION 1 - PRE-ACCIDENT SECTION (Frames 1-{video_metadata.accident_frame-1}):
Use standard XML tags: <road>, <car>, <motorcycle>, <building>, <tree>, etc.

SECTION 2 - ACCIDENT AND RECOVERY SECTION (Frame {video_metadata.accident_frame} onwards):
Use SPECIAL ACCIDENT TAGS for involved objects:
- <accident_object_1>primary object</accident_object_1>
- <accident_object_2>affected object</accident_object_2>
- <accident_object_3>other affected</accident_object_3>
- <accident_object_4>debris/secondary</accident_object_4>

FORMAT:

PRE-ACCIDENT SECTION (Normal Tags):
----------------------------------------
[Describe normal conditions using standard tags]

At **

ACCIDENT AND RECOVERY SECTION (Accident Tags):
----------------------------------------
Frame {video_metadata.accident_frame}**, the accident occurs: [Use accident_object tags]

Generate the caption now:"""
        
        return prompt
    
    def parse_grounded_captions(self, response: str) -> Dict:
        """Parse the response to extract tagged objects"""
        pattern = r'<([\w_]+\d*)>(.*?)</\1>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        grounded_objects = []
        object_counts = {}
        accident_objects = []
        normal_objects = []
        
        for tag, description in matches:
            obj = {
                "object_type": tag,
                "description": description.strip()
            }
            grounded_objects.append(obj)
            
            if tag.startswith("accident_object"):
                accident_objects.append(obj)
            else:
                normal_objects.append(obj)
            
            if tag not in object_counts:
                object_counts[tag] = 0
            object_counts[tag] += 1
        
        has_pre_accident = "PRE-ACCIDENT SECTION" in response
        has_accident_section = "ACCIDENT AND RECOVERY SECTION" in response
        
        return {
            "full_caption": response,
            "objects": grounded_objects,
            "normal_objects": normal_objects,
            "accident_objects": accident_objects,
            "object_counts": object_counts,
            "unique_object_types": list(object_counts.keys()),
            "has_two_sections": has_pre_accident and has_accident_section,
            "total_accident_objects": len(accident_objects),
            "total_normal_objects": len(normal_objects)
        }
    
    # ========== ACCIDENT OBJECT TRACKING METHODS ==========
    
    def detect_accident_objects_in_frame(self, 
                                        image_path: Path,
                                        frame_num: int,
                                        accident_frame: int) -> List[Dict]:
        """Detect accident objects in a single frame"""
        prompt = f"""Analyzing frame {frame_num} (accident at frame {accident_frame}).
Detect ONLY accident-related objects:
- accident_object_1: primary accident object (e.g., opening door, detached tire)
- accident_object_2: affected vehicle/motorcycle
- accident_object_3: other affected vehicles
- accident_object_4: debris or secondary objects

Return JSON array:
[{{"bbox_2d": [x1,y1,x2,y2], "object_label": "accident_object_1", "type": "description", "is_accident_related": true}}]"""
        
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
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
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
            
            # Parse JSON response
            clean_response = response
            if "```json" in clean_response:
                clean_response = re.sub(r'```json\s*', '', clean_response)
                clean_response = re.sub(r'```\s*', '', clean_response)
            clean_response = clean_response.strip()
            
            if clean_response.startswith('['):
                return json.loads(clean_response)
            return []
            
        except Exception as e:
            print(f"Error detecting objects: {e}")
            return []
    
    def generate_tracking_points_from_detections(self,
                                                image: np.ndarray,
                                                detections: List[Dict],
                                                points_per_object: int = 10) -> Tuple[np.ndarray, List[Dict]]:
        """Generate tracking points from detected accident objects using SAM"""
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
                
                # Add corner and center points for better tracking
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                corner_points = [
                    [cx, cy],  # Center
                    [bbox[0], bbox[1]],  # Top-left
                    [bbox[2], bbox[3]],  # Bottom-right
                ]
                
                for point in corner_points[:min(3, points_per_object // 3)]:
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
                                          method: str = "expanded") -> Dict[int, Dict[str, np.ndarray]]:
        """Compute bounding boxes from tracked points"""
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
                    
                    # Compute bounding box
                    x_min, y_min = visible_points.min(axis=0)
                    x_max, y_max = visible_points.max(axis=0)
                    
                    if method == "expanded":
                        # Expand by 20% on each side
                        width = x_max - x_min
                        height = y_max - y_min
                        margin_x = width * 0.2
                        margin_y = height * 0.2
                        
                        x_min -= margin_x
                        x_max += margin_x
                        y_min -= margin_y
                        y_max += margin_y
                    
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
    
    def visualize_tracking_with_bboxes(self,
                                      tracking_results: Dict,
                                      output_dir: Path,
                                      save_video: bool = True,
                                      show_points: bool = False,
                                      show_trails: bool = False,
                                      fps: int = 10):
        """Visualize tracking results with bounding boxes for ALL frames"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tracks = tracking_results["tracks"]
        visibility = tracking_results["visibility"]
        point_metadata = tracking_results["point_metadata"]
        frame_paths = tracking_results["frame_paths"]
        frame_bboxes = tracking_results["frame_bboxes"]
        
        # Handle visibility shape
        if len(visibility.shape) == 3:
            visibility = visibility.squeeze(-1)
        
        frames_with_tracks = []
        
        print(f"Creating visualizations for ALL {len(frame_paths)} frames...")
        
        for t in tqdm(range(len(frame_paths)), desc="Visualizing frames"):
            # Load frame
            frame = cv2.imread(frame_paths[t])
            if frame is None:
                continue
            
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
                        color = (0, 255, 0)
                    
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
                        
                        cv2.circle(overlay, (int(point[0]), int(point[1])), 3, color, -1)
            
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
            
            # Save frame as both jpg and png
            frame_path_jpg = output_dir / f"frame_{frame_num:06d}.jpg"
            frame_path_png = output_dir / f"frame_{frame_num:06d}.png"
            cv2.imwrite(str(frame_path_jpg), overlay)
            cv2.imwrite(str(frame_path_png), overlay)
            
            # Collect for video
            if save_video:
                frames_with_tracks.append(overlay)
        
        # Create videos if requested
        if save_video and frames_with_tracks:
            print(f"Creating video from {len(frames_with_tracks)} frames...")
            
            # Video with bounding boxes
            video_path_bbox = output_dir / "tracking_video_with_bboxes.mp4"
            height, width = frames_with_tracks[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path_bbox), fourcc, fps, (width, height))
            
            for frame in frames_with_tracks:
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Video with bboxes saved to {video_path_bbox}")
            
            # Also create a basic tracking video
            video_path_basic = output_dir / "tracking_video.mp4"
            video_writer_basic = cv2.VideoWriter(str(video_path_basic), fourcc, fps, (width, height))
            
            for frame in frames_with_tracks:
                video_writer_basic.write(frame)
            
            video_writer_basic.release()
            print(f"Basic video saved to {video_path_basic}")
        
        print(f"Visualization complete: {len(frames_with_tracks)} frames processed")
    
    def process_video_complete(self,
                              video_folder: str,
                              output_base_dir: str = "accident_analysis_output",
                              generate_caption: bool = True,
                              track_objects: bool = True,
                              visualize: bool = True,
                              use_all_frames_for_tracking: bool = True) -> Dict:
        """
        Complete pipeline with proper folder structure
        
        Args:
            use_all_frames_for_tracking: If True, uses ALL frames for tracking/visualization
        """
        video_folder = Path(video_folder)
        video_id = video_folder.name
        output_dir = Path(output_base_dir) / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING VIDEO: {video_id}")
        print(f"{'='*70}")
        
        # Load metadata
        video_metadata = self.load_video_metadata(video_folder)
        
        print(f"\nVideo Information:")
        print(f"  - Total frames: {video_metadata.total_frames}")
        print(f"  - Conditions: {video_metadata.get_light_str()}, {video_metadata.get_weather_str()}")
        print(f"  - Location: {video_metadata.get_scene_str()} {video_metadata.get_road_str()}")
        
        if video_metadata.accident:
            print(f"  - Accident frame: {video_metadata.accident_frame}")
            print(f"  - Cause: {video_metadata.causes}")
        
        results = {
            "video_id": video_id,
            "metadata": video_metadata,
            "output_dir": str(output_dir)
        }
        
        # Get ALL frames for tracking/visualization
        all_frame_paths = self.get_all_frame_paths(video_folder)
        print(f"\nFound {len(all_frame_paths)} total frames in video")
        
        # Step 1: Generate grounded caption (using selected representative frames)
        if generate_caption:
            print(f"\n{'='*60}")
            print("STEP 1: Generating Grounded Caption")
            print(f"{'='*60}")
            
            # Create grounded_captions subdirectory
            caption_dir = output_dir / "grounded_captions"
            caption_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Get available frames
                available_frames = self.get_available_frames(video_folder)
                if not available_frames:
                    raise ValueError(f"No frame files found in {video_folder}")
                
                # Select REPRESENTATIVE frames for caption (to avoid API limits)
                selected_frames = self.select_representative_frames(
                    video_metadata, available_frames, max_frames=20
                )
                
                frame_numbers = [num for num, _ in selected_frames]
                print(f"Selected {len(selected_frames)} representative frames for caption: {frame_numbers}")
                
                # Prepare API call
                frames_for_api = []
                frames_for_api.append({
                    "text": f"Analyzing {len(selected_frames)} frames from a {video_metadata.total_frames}-frame video.\n"
                           f"Frame numbers: {frame_numbers}\n"
                })
                
                for frame_num, frame_path in selected_frames:
                    if frame_num < video_metadata.accident_frame:
                        section = "PRE-ACCIDENT"
                    else:
                        section = "ACCIDENT/RECOVERY"
                    
                    frames_for_api.append({"text": f"\nFrame {frame_num} ({section}):"})
                    frames_for_api.append({"image_path": str(frame_path)})
                
                prompt = self.create_comprehensive_grounding_prompt(video_metadata, len(selected_frames))
                frames_for_api.append({"text": prompt})
                
                # Call API
                print("Calling DASHSCOPE API...")
                if self.use_openai_client:
                    content = []
                    for item in frames_for_api:
                        if "text" in item:
                            content.append({"type": "text", "text": item["text"]})
                        elif "image_path" in item:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": self.create_local_image_url(item["image_path"])}
                            })
                    
                    completion = self.client.chat.completions.create(
                        model="qwen-vl-max",
                        messages=[{"role": "user", "content": content}],
                        max_tokens=4000,
                        temperature=0.1
                    )
                    response = completion.choices[0].message.content
                else:
                    response = self.call_api_with_requests_multi_image(
                        frames_for_api, model="qwen-vl-max", max_tokens=4000, temperature=0.1
                    )
                
                # Parse response
                grounded_output = self.parse_grounded_captions(response)
                
                # Save grounded caption
                caption_result = {
                    "video_id": video_id,
                    "grounded_caption": grounded_output,
                    "frame_numbers_used": frame_numbers
                }
                
                with open(caption_dir / f"{video_id}_grounded_caption.json", 'w') as f:
                    json.dump(caption_result, f, indent=2)
                
                with open(caption_dir / f"{video_id}_grounded_caption.txt", 'w') as f:
                    f.write(grounded_output["full_caption"])
                
                results["grounded_caption"] = caption_result
                print("✓ Grounded caption generated successfully")
                
            except Exception as e:
                print(f"✗ Failed to generate grounded caption: {e}")
        
        # Step 2: Track accident objects (using ALL frames)
        if track_objects and video_metadata.accident and self.enable_tracking:
            print(f"\n{'='*60}")
            print("STEP 2: Tracking Accident Objects")
            print(f"{'='*60}")
            
            try:
                # Detect objects in accident frames
                initial_detections = {}
                for offset in range(3):
                    frame_num = video_metadata.accident_frame + offset
                    # Find the frame path
                    frame_path = None
                    for fn, fp in all_frame_paths:
                        if fn == frame_num:
                            frame_path = fp
                            break
                    
                    if frame_path and frame_path.exists():
                        print(f"Detecting objects in frame {frame_num}...")
                        detections = self.detect_accident_objects_in_frame(
                            frame_path, frame_num, video_metadata.accident_frame
                        )
                        initial_detections[frame_num] = detections
                        print(f"  Found {len(detections)} accident objects")
                
                # Save initial detections
                with open(output_dir / "initial_detections.json", 'w') as f:
                    json.dump({str(k): v for k, v in initial_detections.items()}, f, indent=2)
                
                if not any(initial_detections.values()):
                    print("No accident objects detected")
                else:
                    # Load ALL frames for tracking
                    if use_all_frames_for_tracking:
                        # Use ALL frames
                        start_frame = 1
                        end_frame = video_metadata.total_frames
                        print(f"Using ALL frames ({start_frame} to {end_frame}) for tracking")
                    else:
                        # Use limited range around accident
                        start_frame = max(1, video_metadata.accident_frame - 10)
                        end_frame = min(video_metadata.accident_frame + 50, video_metadata.total_frames)
                        print(f"Using frames {start_frame} to {end_frame} for tracking")
                    
                    frames_list = []
                    frame_paths = []
                    
                    # Load all frames in the range
                    for frame_num in range(start_frame, end_frame + 1):
                        # Find the frame path
                        frame_path = None
                        for fn, fp in all_frame_paths:
                            if fn == frame_num:
                                frame_path = fp
                                break
                        
                        if frame_path and frame_path.exists():
                            frame = cv2.imread(str(frame_path))
                            if frame is not None:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames_list.append(frame)
                                frame_paths.append(str(frame_path))
                    
                    print(f"Loaded {len(frames_list)} frames for tracking")
                    
                    if len(frames_list) >= 2:
                        # Convert to tensor
                        video_tensor = torch.tensor(np.stack(frames_list)).permute(0, 3, 1, 2).float()
                        video_tensor = video_tensor.unsqueeze(0).to(self.device)
                        
                        # Generate tracking points
                        all_queries = []
                        all_point_metadata = []
                        
                        for frame_num, detections in initial_detections.items():
                            if not detections:
                                continue
                            
                            frame_idx = frame_num - start_frame
                            if frame_idx >= len(frames_list) or frame_idx < 0:
                                continue
                            
                            # Generate points from detections
                            points, metadata = self.generate_tracking_points_from_detections(
                                frames_list[frame_idx],
                                detections,
                                points_per_object=8
                            )
                            
                            if len(points) > 0:
                                # Add time dimension to points
                                points_with_time = np.column_stack([
                                    np.full(len(points), frame_idx),
                                    points
                                ])
                                all_queries.append(points_with_time)
                                all_point_metadata.extend(metadata)
                        
                        if all_queries:
                            queries = np.vstack(all_queries)
                            queries_tensor = torch.tensor(queries).float().to(self.device)
                            
                            print(f"Tracking {len(queries)} points across {len(frames_list)} frames...")
                            
                            # Run CoTracker
                            pred_tracks, pred_visibility = self.cotracker(
                                video_tensor,
                                queries=queries_tensor[None]
                            )
                            
                            # Convert results
                            pred_tracks = pred_tracks[0].cpu().numpy()
                            pred_visibility = pred_visibility[0].cpu().numpy()
                            
                            if len(pred_visibility.shape) == 3:
                                pred_visibility = pred_visibility.squeeze(-1)
                            
                            # Compute bounding boxes
                            frame_bboxes = self.compute_bounding_boxes_from_tracks(
                                pred_tracks,
                                pred_visibility,
                                all_point_metadata,
                                method="expanded"
                            )
                            
                            # Save tracking data
                            tracking_dir = output_dir / "tracking_data"
                            tracking_dir.mkdir(parents=True, exist_ok=True)
                            
                            np.save(tracking_dir / "tracks.npy", pred_tracks)
                            np.save(tracking_dir / "visibility.npy", pred_visibility)
                            
                            # Save bounding boxes
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
                            
                            with open(tracking_dir / "bounding_boxes.json", 'w') as f:
                                json.dump(bbox_data, f, indent=2)
                            
                            # Save metadata
                            with open(tracking_dir / "tracking_metadata.json", 'w') as f:
                                json.dump({
                                    "video_id": video_id,
                                    "start_frame": start_frame,
                                    "end_frame": end_frame,
                                    "num_frames": len(frames_list),
                                    "num_tracked_points": len(queries),
                                    "point_metadata": all_point_metadata,
                                    "frame_paths": frame_paths
                                }, f, indent=2)
                            
                            # Create tracking results for visualization
                            tracking_results = {
                                "tracks": pred_tracks,
                                "visibility": pred_visibility,
                                "point_metadata": all_point_metadata,
                                "frame_paths": frame_paths,
                                "frame_bboxes": frame_bboxes,
                                "start_frame": start_frame,
                                "end_frame": end_frame
                            }
                            
                            results["tracking"] = tracking_results
                            print(f"✓ Tracking completed successfully for {len(frames_list)} frames")
                            
                            # Visualize if requested
                            if visualize:
                                print(f"\n{'='*60}")
                                print("STEP 3: Creating Visualizations")
                                print(f"{'='*60}")
                                
                                viz_dir = output_dir / "visualizations"
                                self.visualize_tracking_with_bboxes(
                                    tracking_results,
                                    viz_dir,
                                    save_video=True,
                                    show_points=False,
                                    show_trails=False,
                                    fps=10
                                )
                                print("✓ Visualizations created successfully")
                        
            except Exception as e:
                print(f"✗ Failed to track accident objects: {e}")
                import traceback
                traceback.print_exc()
        
        # Save combined analysis results
        with open(output_dir / "complete_analysis.json", 'w') as f:
            json_safe_results = {
                "video_id": video_id,
                "metadata": {
                    "video_id": video_metadata.video_id,
                    "total_frames": video_metadata.total_frames,
                    "accident_frame": video_metadata.accident_frame,
                    "causes": video_metadata.causes,
                    "weather": video_metadata.get_weather_str(),
                    "light": video_metadata.get_light_str(),
                    "scene": video_metadata.get_scene_str(),
                    "road": video_metadata.get_road_str()
                },
                "output_dir": str(output_dir),
                "components": {
                    "grounded_caption": "grounded_caption" in results,
                    "tracking": "tracking" in results,
                    "visualizations": visualize and "tracking" in results
                },
                "frames_processed": len(all_frame_paths)
            }
            json.dump(json_safe_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✓ PROCESSING COMPLETE")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Total frames processed: {len(all_frame_paths)}")
        print(f"\nFolder structure:")
        print(f"  {output_dir}/")
        print(f"  ├── initial_detections.json")
        print(f"  ├── complete_analysis.json")
        if generate_caption:
            print(f"  ├── grounded_captions/")
            print(f"  │   ├── {video_id}_grounded_caption.json")
            print(f"  │   └── {video_id}_grounded_caption.txt")
        if track_objects and video_metadata.accident:
            print(f"  ├── tracking_data/")
            print(f"  │   ├── bounding_boxes.json")
            print(f"  │   ├── tracking_metadata.json")
            print(f"  │   ├── tracks.npy")
            print(f"  │   └── visibility.npy")
            if visualize:
                print(f"  └── visualizations/")
                print(f"      ├── frame_*.jpg ({len(all_frame_paths)} frames)")
                print(f"      ├── frame_*.png ({len(all_frame_paths)} frames)")
                print(f"      ├── tracking_video.mp4")
                print(f"      └── tracking_video_with_bboxes.mp4")
        print(f"{'='*70}")
        
        return results


def main():
    """Main function to run the complete accident video analysis"""
    
    # Configuration
    BASE_DIR = "."
    VIDEO_FOLDER = "000909"  # Video with door-opening accident
    
    # Check for API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable")
        api_key = input("Or enter your DASHSCOPE API key: ").strip()
    
    # SAM checkpoint path (optional, for tracking)
    sam_checkpoint = f"{BASE_DIR}/sam_vit_h_4b8939.pth"
    enable_tracking = os.path.exists(sam_checkpoint)
    
    if not enable_tracking:
        print(f"Warning: SAM checkpoint not found at {sam_checkpoint}")
        print("Tracking and visualization functionality will be disabled")
    
    # Initialize analyzer
    print("\n" + "="*70)
    print("INITIALIZING ACCIDENT VIDEO ANALYZER")
    print("="*70)
    
    analyzer = AccidentVideoAnalyzer(
        api_key=api_key,
        sam_checkpoint=sam_checkpoint if enable_tracking else None,
        sam_model_type="vit_h",
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_tracking=enable_tracking
    )
    
    # Process video
    print("\n" + "="*70)
    print(f"ANALYZING VIDEO: {VIDEO_FOLDER}")
    print("="*70)
    
    try:
        results = analyzer.process_video_complete(
            video_folder=VIDEO_FOLDER,
            output_base_dir="accident_analysis_output_new_on_entire_video",
            generate_caption=True,
            track_objects=enable_tracking,
            visualize=enable_tracking,
            use_all_frames_for_tracking=True  # Use ALL frames for tracking/visualization
        )
        
        print("\n🎉 SUCCESS! Analysis complete.")
        print("\nGenerated outputs:")
        print("1. Grounded caption with two sections (pre-accident and accident)")
        print("2. Initial accident object detections")
        if enable_tracking:
            print("3. Accident object tracking with bounding boxes for ALL frames")
            print("4. Visualization frames and videos containing ALL frames")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()