import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import requests
from datetime import datetime

class AccidentObjectDetector:
    def __init__(self, api_key: str = None, image_dir: str = "", json_path: str = "data.json"):
        """
        Initialize the Accident Object Detector for frame-by-frame detection
        
        Args:
            api_key: API key for DashScope/Model Studio
            image_dir: Directory containing video frame folders
            json_path: Path to the JSON file with video metadata
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        
        # Try to use OpenAI client, but fall back to requests if it fails
        self.use_openai_client = False
        try:
            from openai import OpenAI
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self.use_openai_client = True
            except TypeError:
                try:
                    import openai
                    openai.api_key = self.api_key
                    openai.api_base = self.base_url
                    self.use_openai_client = False
                except:
                    self.use_openai_client = False
        except ImportError:
            print("OpenAI library not found, using requests instead")
            self.use_openai_client = False
        
        self.image_dir = Path(image_dir)
        self.json_path = Path(json_path)
        
        # Mapping for metadata interpretation
        self.weather_map = {1: "sunny", 2: "rainy", 3: "snowy", 4: "foggy"}
        self.light_map = {1: "day", 2: "night"}
        self.scene_map = {1: "highway", 2: "tunnel", 3: "mountain", 4: "urban", 5: "rural"}
        self.road_map = {1: "arterials", 2: "curve", 3: "intersection", 4: "T-junction", 5: "ramp"}
    
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_local_image_url(self, image_path: Path) -> str:
        """Create a data URL for local image"""
        base64_image = self.encode_image_to_base64(image_path)
        mime_type = "image/jpeg" if image_path.suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
        return f"data:{mime_type};base64,{base64_image}"
    
    def call_api_with_requests(self, message_content, model="qwen-vl-max", max_tokens=2000, temperature=0.1):
        """Call API using requests library as fallback"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")
    
    def parse_metadata(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON entry into structured metadata"""
        return {
            "weather": self.weather_map.get(entry.get("weather(sunny,rainy,snowy,foggy)1-4", 1)),
            "lighting": self.light_map.get(entry.get("light(day,night)1-2", 1)),
            "scene": self.scene_map.get(entry.get("scenes(highway,tunnel,mountain,urban,rural)1-5", 4)),
            "road_type": self.road_map.get(entry.get("linear(arterials,curve,intersection,T-junction,ramp) 1-5", 1)),
            "accident_occurred": entry.get('whether an accident occurred (1/0)', 0) == 1,
            "abnormal_start_frame": entry.get('abnormal start frame', None),
            "abnormal_end_frame": entry.get('abnormal end frame', None),
            "accident_frame": entry.get('accident frame', None),
            "total_frames": entry.get('total frames', None),
            "event_description": entry.get('texts', ''),
            "cause": entry.get('causes', ''),
            "measures": entry.get('measures', '')
        }
    
    def create_accident_detection_prompt(self, frame_number: int, accident_frame: int, 
                                        dense_caption: str = None) -> str:
        """Create prompt specifically for accident object detection"""
        
        # Extract accident information from dense caption if provided
        accident_context = ""
        if dense_caption and "accident_object" in dense_caption.lower():
            # Extract accident object descriptions
            accident_patterns = re.findall(r'<accident_object_\d+>(.*?)</accident_object_\d+>', dense_caption)
            if accident_patterns:
                accident_context = "\nKnown accident objects to look for:\n"
                for i, desc in enumerate(accident_patterns[:4], 1):
                    accident_context += f"- accident_object_{i}: {desc}\n"
        
        prompt = f"""You are an expert accident detection system analyzing frame {frame_number:06d} of a traffic accident sequence.
The accident occurred at frame {accident_frame:06d}.{"" if frame_number == accident_frame else f" This is frame {frame_number - accident_frame:+d} relative to the accident."}

{accident_context}

Analyze this frame and detect ALL objects, with special attention to accident-related items:

For EACH object detected, provide information in this exact JSON format:

[
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "object_label": "accident_object_1",
    "type": "detached_tire",
    "description": "black tire rolling across lanes",
    "is_accident_related": true
  }},
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "object_label": "accident_object_2",
    "type": "damaged_vehicle",
    "description": "sedan with missing rear tire",
    "is_accident_related": true
  }},
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "object_label": "vehicle",
    "type": "car",
    "description": "uninvolved vehicle in left lane",
    "is_accident_related": false
  }}
]

ACCIDENT OBJECTS to identify (use labels accident_object_1 through accident_object_4):
- Detached parts (tires, wheels, bumpers, debris)
- Damaged/affected vehicles
- Vehicles taking evasive action
- Debris/fragments on road

NORMAL OBJECTS to detect:
- Vehicles: car, truck, bus, motorcycle, bicycle, van, SUV
- People: pedestrian, cyclist, person
- Infrastructure: traffic_sign, traffic_light, street_light, barrier, guardrail, pole, building
- Road elements: road, lane_marking, crosswalk, sidewalk, curb

Requirements:
1. bbox_2d must be [left, top, right, bottom] in pixel coordinates
2. For accident-related objects, use "accident_object_N" as object_label (N=1,2,3,4)
3. Set is_accident_related to true for any object involved in or affected by the accident
4. Provide detailed descriptions especially for accident objects

IMPORTANT: Return ONLY the JSON array, no other text. Start with [ and end with ]"""
        
        return prompt
    
    def parse_detection_response(self, response: str, frame_name: str) -> List[Dict]:
        """Parse the model's response to extract detected objects"""
        detected_objects = []
        
        print(f"    Response preview: {response[:500] if len(response) > 500 else response}")
        
        try:
            # Clean up response
            clean_response = response
            
            # Remove markdown code blocks
            if "```json" in clean_response:
                clean_response = re.sub(r'```json\s*', '', clean_response)
                clean_response = re.sub(r'```\s*', '', clean_response)
            elif "```" in clean_response:
                clean_response = re.sub(r'```\s*', '', clean_response)
            
            clean_response = clean_response.strip()
            
            # Try to parse as JSON array
            if clean_response.startswith('['):
                parsed_objects = json.loads(clean_response)
                
                for obj in parsed_objects:
                    if isinstance(obj, dict) and 'bbox_2d' in obj:
                        cleaned_obj = {
                            "frame": frame_name,
                            "bbox_2d": obj.get("bbox_2d", [0, 0, 0, 0]),
                            "object_label": obj.get("object_label", "unknown"),
                            "type": obj.get("type", "unknown"),
                            "description": obj.get("description", ""),
                            "is_accident_related": obj.get("is_accident_related", False)
                        }
                        detected_objects.append(cleaned_obj)
                
                return detected_objects
            
        except json.JSONDecodeError as e:
            print(f"    Warning: Could not parse JSON from response for frame {frame_name}: {e}")
        except Exception as e:
            print(f"    Error parsing response: {e}")
        
        if not detected_objects:
            print(f"    Warning: No valid objects parsed from response")
        
        return detected_objects
    
    def get_video_frames(self, video_folder: str) -> List[Path]:
        """Get all frames from video folder"""
        video_folder_path = Path(video_folder)
        
        # Try different path combinations
        possible_paths = [
            video_folder_path / "images",
            self.image_dir / video_folder_path / "images",
            self.image_dir / video_folder / "images",
            self.image_dir / video_folder,
            video_folder_path
        ]
        
        folder_path = None
        for path in possible_paths:
            if path.exists():
                folder_path = path
                break
        
        if folder_path is None:
            print(f"Warning: Video folder not found in any of: {possible_paths}")
            return []
        
        print(f"Looking for frames in: {folder_path}")
        
        # Get all image files
        frames = sorted([
            f for f in folder_path.iterdir() 
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        
        print(f"Found {len(frames)} frames in {folder_path}")
        return frames
    
    def get_frame_by_number(self, frames: List[Path], frame_number: int) -> Optional[Path]:
        """Get specific frame by frame number"""
        frame_name = f"{frame_number:06d}.jpg"
        for frame in frames:
            if frame.name == frame_name:
                return frame
        return None
    
    def detect_accident_objects_in_frame(self, image_path: Path, frame_number: int, 
                                        accident_frame: int, dense_caption: str = None) -> List[Dict]:
        """Detect objects in a single frame with accident context"""
        prompt = self.create_accident_detection_prompt(frame_number, accident_frame, dense_caption)
        
        try:
            # Create message content with image and prompt
            image_url = self.create_local_image_url(image_path)
            message_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            # Make API call
            if self.use_openai_client:
                completion = self.client.chat.completions.create(
                    model="qwen-vl-max",
                    messages=[{"role": "user", "content": message_content}],
                    max_tokens=2000,
                    temperature=0.1
                )
                response = completion.choices[0].message.content
            else:
                response = self.call_api_with_requests(
                    message_content,
                    model="qwen-vl-max",
                    max_tokens=2000,
                    temperature=0.1
                )
            
            detected_objects = self.parse_detection_response(response, image_path.name)
            
            # Add frame context
            for obj in detected_objects:
                obj["frame_number"] = frame_number
                obj["frames_from_accident"] = frame_number - accident_frame
            
            return detected_objects
            
        except Exception as e:
            print(f"    Error detecting objects in {image_path.name}: {e}")
            return []
    
    def save_accident_detection(self, video_folder: str, frame_name: str, frame_number: int,
                               detected_objects: List[Dict], output_dir: str = "accident_detections"):
        """Save accident detection results for a single frame"""
        # Create output directory structure
        output_path = Path(output_dir) / video_folder / "accident_frames"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Count accident-related objects
        accident_objects = [obj for obj in detected_objects if obj.get("is_accident_related", False)]
        
        # Create frame result
        frame_result = {
            "video": video_folder,
            "frame": frame_name,
            "frame_number": frame_number,
            "timestamp": datetime.now().isoformat(),
            "total_objects": len(detected_objects),
            "accident_objects": len(accident_objects),
            "objects": detected_objects
        }
        
        # Save individual frame detection
        frame_file = output_path / f"frame_{frame_number:06d}_detection.json"
        with open(frame_file, 'w') as f:
            json.dump(frame_result, f, indent=2, ensure_ascii=False)
        
        print(f"    Saved accident detection to: {frame_file}")
        
        return frame_result
    
    def detect_accident_sequence(self, entry: Dict[str, Any], 
                                dense_caption: str = None,
                                frames_before: int = 5, 
                                frames_after: int = 20,
                                output_dir: str = "accident_detections") -> Dict[str, Any]:
        """
        Detect objects in accident sequence frames
        
        Args:
            entry: JSON entry with video metadata
            dense_caption: Dense caption with accident object descriptions
            frames_before: Number of frames before accident to process
            frames_after: Number of frames after accident to process
            output_dir: Directory to save detection results
        
        Returns:
            Dictionary with accident sequence detections
        """
        video_folder = entry.get("Video", "")
        metadata = self.parse_metadata(entry)
        
        if not metadata.get("accident_occurred"):
            print(f"No accident in video {video_folder}, skipping...")
            return {"video": video_folder, "accident": False}
        
        accident_frame = metadata.get("accident_frame")
        if accident_frame is None:
            print(f"No accident frame specified for {video_folder}, skipping...")
            return {"video": video_folder, "accident": False, "error": "No accident frame specified"}
        
        accident_frame = int(accident_frame)
        
        print(f"\n{'='*60}")
        print(f"Processing accident sequence for video: {video_folder}")
        print(f"  Accident frame: {accident_frame}")
        print(f"  Processing frames: {accident_frame - frames_before} to {accident_frame + frames_after}")
        print(f"  Scene: {metadata['scene']}, Weather: {metadata['weather']}, Lighting: {metadata['lighting']}")
        print(f"  Cause: {metadata['cause']}")
        print(f"{'='*60}")
        
        # Create output directory for this video
        video_output_dir = Path(output_dir) / video_folder
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = video_output_dir / "accident_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "video": video_folder,
                "metadata": metadata,
                "accident_frame": accident_frame,
                "frames_before": frames_before,
                "frames_after": frames_after,
                "processing_started": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # Get all frames
        frames = self.get_video_frames(video_folder)
        
        if not frames:
            print(f"  Warning: No frames found for {video_folder}")
            return {"video": video_folder, "accident": True, "error": "No frames found"}
        
        # Determine frame range to process
        start_frame = max(1, accident_frame - frames_before)
        end_frame = min(len(frames), accident_frame + frames_after)
        
        print(f"\nProcessing frames {start_frame} to {end_frame}...")
        
        all_detected_objects = []
        accident_frame_results = []
        accident_object_tracking = {}  # Track accident objects across frames
        
        # Create progress file
        progress_file = video_output_dir / "accident_progress.json"
        
        # Process each frame in the accident sequence
        for frame_num in range(start_frame, end_frame + 1):
            frame = self.get_frame_by_number(frames, frame_num)
            if not frame:
                print(f"  Frame {frame_num:06d} not found, skipping...")
                continue
            
            relative_position = frame_num - accident_frame
            position_str = "AT ACCIDENT" if relative_position == 0 else f"{relative_position:+d} from accident"
            print(f"\n  Frame {frame_num:06d} ({position_str}):")
            
            # Detect objects in this frame
            detected_objects = self.detect_accident_objects_in_frame(
                frame, frame_num, accident_frame, dense_caption
            )
            
            # Save frame detection immediately
            frame_result = self.save_accident_detection(
                video_folder, 
                frame.name, 
                frame_num,
                detected_objects,
                output_dir
            )
            accident_frame_results.append(frame_result)
            
            if detected_objects:
                all_detected_objects.extend(detected_objects)
                
                # Track accident objects
                accident_objs = [obj for obj in detected_objects if obj.get("is_accident_related", False)]
                for obj in accident_objs:
                    label = obj.get("object_label", "unknown")
                    if label not in accident_object_tracking:
                        accident_object_tracking[label] = []
                    accident_object_tracking[label].append({
                        "frame": frame_num,
                        "type": obj.get("type"),
                        "description": obj.get("description"),
                        "bbox": obj.get("bbox_2d")
                    })
                
                print(f"    Detected {len(detected_objects)} objects ({len(accident_objs)} accident-related)")
            else:
                print(f"    No objects detected")
            
            # Update progress
            frames_processed = frame_num - start_frame + 1
            total_frames_to_process = end_frame - start_frame + 1
            progress_data = {
                "video": video_folder,
                "accident_frame": accident_frame,
                "current_frame": frame_num,
                "frames_processed": frames_processed,
                "total_frames_to_process": total_frames_to_process,
                "progress_percentage": (frames_processed / total_frames_to_process) * 100,
                "accident_objects_found": len(accident_object_tracking),
                "total_objects_detected": len(all_detected_objects),
                "last_updated": datetime.now().isoformat()
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        # Create summary statistics
        accident_objects_count = sum(1 for obj in all_detected_objects if obj.get("is_accident_related", False))
        
        detection_summary = {
            "frames_processed": end_frame - start_frame + 1,
            "total_objects_detected": len(all_detected_objects),
            "accident_objects_detected": accident_objects_count,
            "accident_object_tracking": accident_object_tracking,
            "unique_accident_objects": list(accident_object_tracking.keys()),
            "processing_completed": datetime.now().isoformat()
        }
        
        # Create final result
        final_result = {
            **entry,
            "metadata": metadata,
            "accident_sequence": {
                "start_frame": start_frame,
                "accident_frame": accident_frame,
                "end_frame": end_frame,
                "frames_before": frames_before,
                "frames_after": frames_after
            },
            "detected_objects": all_detected_objects,
            "accident_object_tracking": accident_object_tracking,
            "detection_summary": detection_summary,
            "frame_results": accident_frame_results
        }
        
        # Save final summary
        summary_file = video_output_dir / "accident_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Completed accident sequence processing for {video_folder}")
        print(f"  Frames processed: {detection_summary['frames_processed']}")
        print(f"  Total objects detected: {detection_summary['total_objects_detected']}")
        print(f"  Accident objects detected: {detection_summary['accident_objects_detected']}")
        print(f"  Unique accident objects: {detection_summary['unique_accident_objects']}")
        print(f"  Results saved to: {summary_file}")
        print(f"{'='*60}\n")
        
        return final_result
    
    def process_accident_videos(self, dense_captions_file: str = None,
                               output_dir: str = "accident_detections",
                               frames_before: int = 5,
                               frames_after: int = 20,
                               max_videos: int = None):
        """
        Process all accident videos from JSON file
        
        Args:
            dense_captions_file: Path to file with dense captions (optional)
            output_dir: Directory to save detection results
            frames_before: Frames before accident to process
            frames_after: Frames after accident to process
            max_videos: Maximum number of videos to process
        """
        # Load JSON data
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Load dense captions if provided
        dense_captions = {}
        if dense_captions_file and Path(dense_captions_file).exists():
            with open(dense_captions_file, 'r') as f:
                caption_data = f.read()
                # Parse dense captions per video if structured
                # For now, we'll use the whole caption
                dense_captions = {"default": caption_data}
        
        # Handle both single entry and list of entries
        if isinstance(data, dict):
            entries = [data]
        else:
            entries = data
        
        # Filter only accident videos
        accident_entries = [e for e in entries if e.get('whether an accident occurred (1/0)', 0) == 1]
        
        if max_videos:
            accident_entries = accident_entries[:max_videos]
        
        total = len(accident_entries)
        print(f"\n{'='*60}")
        print(f"Found {total} accident videos to process")
        print(f"Will process {frames_before} frames before and {frames_after} frames after each accident")
        print(f"Results will be saved to: {output_dir}/")
        print(f"{'='*60}\n")
        
        # Create main output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create overall progress file
        overall_progress_file = Path(output_dir) / "overall_accident_progress.json"
        
        results = []
        
        for idx, entry in enumerate(accident_entries, 1):
            print(f"\n[Video {idx}/{total}]")
            
            # Update overall progress
            overall_progress = {
                "total_accident_videos": total,
                "videos_processed": idx - 1,
                "current_video": entry.get('Video', 'Unknown'),
                "progress_percentage": ((idx - 1) / total) * 100,
                "last_updated": datetime.now().isoformat()
            }
            with open(overall_progress_file, 'w') as f:
                json.dump(overall_progress, f, indent=2, ensure_ascii=False)
            
            # Get dense caption for this video (if available)
            video_caption = dense_captions.get(entry.get('Video', ''), dense_captions.get("default", None))
            
            # Process video accident sequence
            result = self.detect_accident_sequence(
                entry, 
                dense_caption=video_caption,
                frames_before=frames_before,
                frames_after=frames_after,
                output_dir=output_dir
            )
            results.append(result)
            
            # Save cumulative results
            cumulative_file = Path(output_dir) / "cumulative_accident_results.json"
            with open(cumulative_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Update final overall progress
        overall_progress = {
            "total_accident_videos": total,
            "videos_processed": total,
            "progress_percentage": 100,
            "completed_at": datetime.now().isoformat()
        }
        with open(overall_progress_file, 'w') as f:
            json.dump(overall_progress, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Total accident videos processed: {len(results)}")
        
        # Summary statistics
        total_accident_objects = sum(
            r.get('detection_summary', {}).get('accident_objects_detected', 0) 
            for r in results if 'detection_summary' in r
        )
        total_frames = sum(
            r.get('detection_summary', {}).get('frames_processed', 0) 
            for r in results if 'detection_summary' in r
        )
        
        print(f"Total frames processed: {total_frames}")
        print(f"Total accident objects detected: {total_accident_objects}")
        if total_frames > 0:
            print(f"Average accident objects per frame: {total_accident_objects/total_frames:.2f}")
        
        print(f"\nResults saved in: {output_dir}/")
        print(f"  - Individual video folders with frame-by-frame detections")
        print(f"  - cumulative_accident_results.json: All results combined")
        print(f"  - overall_accident_progress.json: Processing progress")
        
        return results


# Example usage
def main():
    # Set your API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable")
        return
    
    # Initialize detector
    detector = AccidentObjectDetector(
        api_key=api_key,
        image_dir=".",  # Current directory or path to videos
        json_path="data.json"  # Your JSON file with accident metadata
    )
    
    # Dense caption text (you can load this from a file)
    dense_caption = """
    At Frame 27, the accident occurs: <accident_object_1>a black tire detaches from the rear of the dark-colored sedan</accident_object_1>, 
    flying off into the roadway and rolling erratically across the lanes. This sudden event marks the transition from normal to abnormal conditions. 
    The <accident_object_2>sedan that lost the tire</accident_object_2> immediately shows signs of instability, swerving slightly as it loses control. 
    The <accident_object_3>green sedan</accident_object_3> in the adjacent lane reacts quickly, braking and shifting to avoid collision with the rolling tire. 
    The <accident_object_4>detached tire</accident_object_4> bounces across the road surface, creating a hazard for other drivers.
    """
    
    # Test with single accident entry
    test_entry = {
        "Video": "002320",
        "weather(sunny,rainy,snowy,foggy)1-4": 1,
        "light(day,night)1-2": 1,
        "scenes(highway,tunnel,mountain,urban,rural)1-5": 4,
        "linear(arterials,curve,intersection,T-junction,ramp) 1-5": 1,
        "whether an accident occurred (1/0)": 1,
        "abnormal start frame": 5,
        "abnormal end frame": 85,
        "accident frame": 27,
        "total frames": 113,
        "texts": "[CLS]there is an object crash[SEP]",
        "causes": "vehicle tires fall off when driving",
        "measures": "Before driving, the driver should visually check whether the bolts and nuts of the wheels are complete"
    }
    
    # Detect accident objects in the accident sequence
    print("Testing accident detection for video 002320...")
    result = detector.detect_accident_sequence(
        test_entry, 
        dense_caption=dense_caption,
        frames_before=5,   # Process 5 frames before accident
        frames_after=20,    # Process 20 frames after accident
        output_dir="accident_detections"
    )
    
    print(f"\n{'='*60}")
    print(f"Detection complete!")
    if 'detection_summary' in result:
        summary = result['detection_summary']
        print(f"Frames processed: {summary['frames_processed']}")
        print(f"Total objects detected: {summary['total_objects_detected']}")
        print(f"Accident objects detected: {summary['accident_objects_detected']}")
        print(f"Unique accident objects: {summary['unique_accident_objects']}")
    
    print(f"\nCheck the following directory for results:")
    print(f"  accident_detections/002320/")
    print(f"  - accident_metadata.json: Video and accident metadata")
    print(f"  - accident_progress.json: Processing progress")
    print(f"  - accident_summary.json: Final summary with all detections")
    print(f"  - accident_frames/: Frame-by-frame detection JSONs")
    
    # To process all accident videos in your JSON file:
    # detector.process_accident_videos(
    #     dense_captions_file="dense_captions.txt",  # Optional
    #     output_dir="accident_detections",
    #     frames_before=5,
    #     frames_after=20,
    #     max_videos=5  # Process first 5 accident videos
    # )


if __name__ == "__main__":
    main()