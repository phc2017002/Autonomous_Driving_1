import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Any
import re
import requests
from datetime import datetime

class ObjectDetector:
    def __init__(self, api_key: str = None, image_dir: str = "", json_path: str = "data.json"):
        """
        Initialize the Object Detector for frame-by-frame detection
        
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
            # Try newer OpenAI client format
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self.use_openai_client = True
            except TypeError:
                # Try without base_url if that's causing issues
                try:
                    import openai
                    openai.api_key = self.api_key
                    openai.api_base = self.base_url
                    self.use_openai_client = False  # Use old-style API
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
    
    def create_detection_prompt(self) -> str:
        """Create prompt for object detection"""
        prompt = """You are an expert object detection system. Analyze this traffic/road scene image and detect ALL objects.

For EACH object you can see in the image, provide detection information in this exact JSON format:

[
  {
    "bbox_2d": [x1, y1, x2, y2],
    "object_label": "vehicle",
    "type": "car",
    "description": "white sedan in left lane"
  },
  {
    "bbox_2d": [x1, y1, x2, y2],
    "object_label": "infrastructure",
    "type": "traffic_sign",
    "description": "speed limit sign on right side"
  }
]

Object categories to detect:
- Vehicles: car, truck, bus, motorcycle, bicycle, van, SUV
- People: pedestrian, cyclist, person
- Infrastructure: traffic_sign, traffic_light, street_light, barrier, guardrail, pole, building
- Road elements: road, lane_marking, crosswalk, sidewalk, curb
- Environment: tree, sky, vegetation

Requirements:
1. bbox_2d must be [left, top, right, bottom] in pixel coordinates
2. object_label must be one of: vehicle/person/infrastructure/road_element/environment
3. type must be the specific object type
4. description should include color, position, and distinguishing features

IMPORTANT: Return ONLY the JSON array, no other text. Start with [ and end with ]"""
        return prompt
    
    def parse_detection_response(self, response: str, frame_name: str) -> List[Dict]:
        """Parse the model's response to extract detected objects"""
        detected_objects = []
        
        # Debug: print first 500 chars of response
        print(f"    Response preview: {response[:500] if len(response) > 500 else response}")
        
        try:
            # First try to find JSON array directly
            # Look for content between [ and ]
            json_match = re.search(r'```math.*?```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                # Clean up any markdown formatting
                json_str = re.sub(r'```json\s*', '', json_str)
                json_str = re.sub(r'```\s*', '', json_str)
                
                # Parse the JSON
                parsed_objects = json.loads(json_str)
                
                # Validate and clean each object
                for obj in parsed_objects:
                    if isinstance(obj, dict) and 'bbox_2d' in obj:
                        cleaned_obj = {
                            "frame": frame_name,
                            "bbox_2d": obj.get("bbox_2d", [0, 0, 0, 0]),
                            "object_label": obj.get("object_label", "unknown"),
                            "type": obj.get("type", "unknown"),
                            "description": obj.get("description", "")
                        }
                        detected_objects.append(cleaned_obj)
                
                return detected_objects
            
            # If no JSON array found, try to parse as direct JSON
            try:
                # Remove any markdown code blocks
                clean_response = re.sub(r'```(?:json)?\s*', '', response)
                clean_response = clean_response.strip()
                
                if clean_response.startswith('['):
                    parsed_objects = json.loads(clean_response)
                    for obj in parsed_objects:
                        if isinstance(obj, dict) and 'bbox_2d' in obj:
                            cleaned_obj = {
                                "frame": frame_name,
                                "bbox_2d": obj.get("bbox_2d", [0, 0, 0, 0]),
                                "object_label": obj.get("object_label", "unknown"),
                                "type": obj.get("type", "unknown"),
                                "description": obj.get("description", "")
                            }
                            detected_objects.append(cleaned_obj)
            except:
                pass
            
            # Last resort: try to find individual JSON objects
            if not detected_objects:
                json_objects = re.findall(r'\{[^{}]*"bbox_2d"[^{}]*\}', response)
                for json_str in json_objects:
                    try:
                        obj = json.loads(json_str)
                        if 'bbox_2d' in obj:
                            obj['frame'] = frame_name
                            detected_objects.append(obj)
                    except:
                        continue
                        
        except json.JSONDecodeError as e:
            print(f"    Warning: Could not parse JSON from response for frame {frame_name}: {e}")
            
            # Try to extract bbox patterns as fallback
            bbox_pattern = r'"bbox_2d":\s*```math([^```]+)```'
            label_pattern = r'"object_label":\s*"([^"]+)"'
            type_pattern = r'"type":\s*"([^"]+)"'
            desc_pattern = r'"description":\s*"([^"]+)"'
            
            bboxes = re.findall(bbox_pattern, response)
            labels = re.findall(label_pattern, response)
            types = re.findall(type_pattern, response)
            descs = re.findall(desc_pattern, response)
            
            for i in range(min(len(bboxes), len(labels))):
                try:
                    bbox = [float(x.strip()) for x in bboxes[i].split(',')]
                    detected_objects.append({
                        "frame": frame_name,
                        "bbox_2d": bbox,
                        "object_label": labels[i] if i < len(labels) else "unknown",
                        "type": types[i] if i < len(types) else "unknown",
                        "description": descs[i] if i < len(descs) else ""
                    })
                except:
                    continue
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
    
    def save_frame_detection(self, video_folder: str, frame_name: str, detected_objects: List[Dict], 
                           output_dir: str = "detections_output"):
        """Save detection results for a single frame immediately"""
        # Create output directory structure
        output_path = Path(output_dir) / video_folder / "frame_detections"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create frame result
        frame_result = {
            "video": video_folder,
            "frame": frame_name,
            "timestamp": datetime.now().isoformat(),
            "num_objects": len(detected_objects),
            "objects": detected_objects
        }
        
        # Save individual frame detection
        frame_file = output_path / f"{frame_name.replace('.jpg', '')}_detection.json"
        with open(frame_file, 'w') as f:
            json.dump(frame_result, f, indent=2, ensure_ascii=False)
        
        print(f"    Saved frame detection to: {frame_file}")
        
        return frame_result
    
    def detect_objects_in_frame(self, image_path: Path) -> List[Dict]:
        """Detect objects in a single frame"""
        prompt = self.create_detection_prompt()
        
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
                # Use OpenAI client
                completion = self.client.chat.completions.create(
                    model="qwen-vl-max",
                    messages=[{"role": "user", "content": message_content}],
                    max_tokens=2000,
                    temperature=0.1
                )
                response = completion.choices[0].message.content
            else:
                # Use requests as fallback
                response = self.call_api_with_requests(
                    message_content,
                    model="qwen-vl-max",
                    max_tokens=2000,
                    temperature=0.1
                )
            
            detected_objects = self.parse_detection_response(response, image_path.name)
            
            return detected_objects
            
        except Exception as e:
            print(f"    Error detecting objects in {image_path.name}: {e}")
            return []
    
    def detect_objects_in_video(self, entry: Dict[str, Any], sample_rate: int = 1, 
                               output_dir: str = "detections_output") -> Dict[str, Any]:
        """
        Detect objects in all frames of a video and save results immediately
        
        Args:
            entry: JSON entry with video metadata
            sample_rate: Process every nth frame (1 = all frames, 2 = every other frame, etc.)
            output_dir: Directory to save detection results
        
        Returns:
            Dictionary with original entry data plus detected objects
        """
        video_folder = entry.get("Video", "")
        metadata = self.parse_metadata(entry)
        
        print(f"\nProcessing video: {video_folder}")
        print(f"  Scene: {metadata['scene']}, Weather: {metadata['weather']}, Lighting: {metadata['lighting']}")
        
        # Create output directory for this video
        video_output_dir = Path(output_dir) / video_folder
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = video_output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "video": video_folder,
                "metadata": metadata,
                "processing_started": datetime.now().isoformat(),
                "sample_rate": sample_rate
            }, f, indent=2, ensure_ascii=False)
        
        # Get all frames
        frames = self.get_video_frames(video_folder)
        
        if not frames:
            print(f"  Warning: No frames found for {video_folder}")
            error_result = {
                **entry,
                "metadata": metadata,
                "detected_objects": [],
                "detection_summary": {
                    "total_frames": 0,
                    "frames_processed": 0,
                    "total_objects_detected": 0,
                    "error": "No frames found"
                }
            }
            # Save error result
            error_file = video_output_dir / "error.json"
            with open(error_file, 'w') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            return error_result
        
        # Sample frames based on sample_rate
        sampled_frames = frames[::sample_rate]
        print(f"  Processing {len(sampled_frames)} out of {len(frames)} frames (sample_rate={sample_rate})")
        
        all_detected_objects = []
        frame_detections = {}
        frame_results = []
        
        # Create progress file
        progress_file = video_output_dir / "progress.json"
        
        # Process each sampled frame
        for idx, frame_path in enumerate(sampled_frames):
            print(f"  Processing frame {idx+1}/{len(sampled_frames)}: {frame_path.name}")
            
            # Detect objects in this frame
            detected_objects = self.detect_objects_in_frame(frame_path)
            
            # Save frame detection immediately
            frame_result = self.save_frame_detection(
                video_folder, 
                frame_path.name, 
                detected_objects,
                output_dir
            )
            frame_results.append(frame_result)
            
            if detected_objects:
                all_detected_objects.extend(detected_objects)
                frame_detections[frame_path.name] = detected_objects
                print(f"    Detected {len(detected_objects)} objects")
            else:
                print(f"    No objects detected")
            
            # Update progress
            progress_data = {
                "video": video_folder,
                "total_frames": len(frames),
                "sampled_frames": len(sampled_frames),
                "frames_processed": idx + 1,
                "progress_percentage": ((idx + 1) / len(sampled_frames)) * 100,
                "objects_detected_so_far": len(all_detected_objects),
                "last_updated": datetime.now().isoformat()
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        # Create summary statistics
        object_counts = {}
        for obj in all_detected_objects:
            obj_type = obj.get("type", "unknown")
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        detection_summary = {
            "total_frames": len(frames),
            "frames_processed": len(sampled_frames),
            "total_objects_detected": len(all_detected_objects),
            "object_type_counts": object_counts,
            "average_objects_per_frame": len(all_detected_objects) / len(sampled_frames) if sampled_frames else 0,
            "processing_completed": datetime.now().isoformat()
        }
        
        # Identify key frames (accident frame, abnormal start/end)
        key_frame_detections = {}
        if metadata.get("accident_frame"):
            accident_frame_name = f"{int(metadata['accident_frame']):06d}.jpg"
            if accident_frame_name in frame_detections:
                key_frame_detections["accident_frame"] = frame_detections[accident_frame_name]
        
        # Create final result
        final_result = {
            **entry,
            "metadata": metadata,
            "detected_objects": all_detected_objects,
            "frame_detections": frame_detections,
            "key_frame_detections": key_frame_detections,
            "detection_summary": detection_summary,
            "frame_results": frame_results
        }
        
        # Save final summary
        summary_file = video_output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Completed processing. Summary saved to: {summary_file}")
        
        return final_result
    
    def process_json_file(self, output_path: str = "object_detections_output.json", 
                         output_dir: str = "detections_output",
                         sample_rate: int = 5, max_entries: int = None):
        """
        Process entire JSON file and detect objects in all videos
        
        Args:
            output_path: Path to save the final output JSON with all detections
            output_dir: Directory to save individual detection results
            sample_rate: Process every nth frame (to reduce API calls)
            max_entries: Maximum number of entries to process (None = all)
        """
        # Load JSON data
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single entry and list of entries
        if isinstance(data, dict):
            entries = [data]
        else:
            entries = data
        
        # Limit entries if specified
        if max_entries:
            entries = entries[:max_entries]
        
        results = []
        total = len(entries)
        
        print(f"Processing {total} entries with sample_rate={sample_rate}...")
        print(f"Results will be saved to: {output_dir}/")
        print("=" * 50)
        
        # Create main output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create overall progress file
        overall_progress_file = Path(output_dir) / "overall_progress.json"
        
        for idx, entry in enumerate(entries, 1):
            print(f"\nProcessing entry {idx}/{total}: {entry.get('Video', 'Unknown')}")
            
            # Update overall progress
            overall_progress = {
                "total_videos": total,
                "videos_processed": idx - 1,
                "current_video": entry.get('Video', 'Unknown'),
                "progress_percentage": ((idx - 1) / total) * 100,
                "last_updated": datetime.now().isoformat()
            }
            with open(overall_progress_file, 'w') as f:
                json.dump(overall_progress, f, indent=2, ensure_ascii=False)
            
            # Process video
            result = self.detect_objects_in_video(entry, sample_rate=sample_rate, output_dir=output_dir)
            results.append(result)
            
            # Save cumulative results periodically
            if idx % 2 == 0 or idx == total:
                with open(f"{output_path}.tmp", 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  Saved cumulative results ({idx}/{total})")
        
        # Save final consolidated results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Update final overall progress
        overall_progress = {
            "total_videos": total,
            "videos_processed": total,
            "current_video": "Complete",
            "progress_percentage": 100,
            "completed_at": datetime.now().isoformat()
        }
        with open(overall_progress_file, 'w') as f:
            json.dump(overall_progress, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 50)
        print(f"Processing complete!")
        print(f"Individual frame detections saved in: {output_dir}/")
        print(f"Consolidated results saved to: {output_path}")
        
        # Print summary statistics
        total_objects = sum(r['detection_summary']['total_objects_detected'] for r in results)
        total_frames_processed = sum(r['detection_summary']['frames_processed'] for r in results)
        
        print(f"\nOverall Statistics:")
        print(f"  Total videos processed: {len(results)}")
        print(f"  Total frames processed: {total_frames_processed}")
        print(f"  Total objects detected: {total_objects}")
        if total_frames_processed > 0:
            print(f"  Average objects per frame: {total_objects/total_frames_processed:.2f}")
        
        return results


# Example usage
def main():
    # Set your API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable")
        return
    
    # Initialize detector
    detector = ObjectDetector(
        api_key=api_key,
        image_dir=".",  # Current directory or path to videos
        json_path="dense_captions_output_1.json"  # Your JSON file (optional for single video test)
    )
    
    # Test with single entry
    test_entry = {
        "Video": "002320",
        "weather(sunny,rainy,snowy,foggy)1-4": 1,
        "light(day,night)1-2": 1,
        "scenes(highway,tunnel,mountain,urban,rural)1-5": 4,
        "linear(arterials,curve,intersection,T-junction,ramp) 1-5": 1,
        "type": 44,
        "whether an accident occurred (1/0)": 1,
        "abnormal start frame": 5,
        "abnormal end frame": 85,
        "accident frame": 27,
        "total frames": 113,
        "texts": "[CLS]there is an object crash[SEP]",
        "causes": "vehicle tires fall off when driving",
        "measures": "Before driving, the driver should visually check whether the bolts and nuts of the wheels are complete"
    }
    
    # Detect objects in video with immediate saving
    print("Testing with video 002320...")
    result = detector.detect_objects_in_video(
        test_entry, 
        sample_rate=20,  # Process every 20th frame
        output_dir="detections_output"
    )
    
    print(f"\nDetection complete!")
    if 'detection_summary' in result:
        print(f"Summary: {result['detection_summary']}")
    
    print(f"\nCheck the following directory for results:")
    print(f"  detections_output/002320/")
    print(f"  - metadata.json: Video metadata")
    print(f"  - progress.json: Processing progress")
    print(f"  - summary.json: Final summary with all detections")
    print(f"  - frame_detections/: Individual frame detection JSONs")


if __name__ == "__main__":
    main()