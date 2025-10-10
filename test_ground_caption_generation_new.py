import cv2
import numpy as np
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Tuple, Optional, Union
import re
import os
from pathlib import Path
from dataclasses import dataclass
from glob import glob

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

class GroundedCaptionGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the generator with DASHSCOPE API configuration
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not provided and not found in environment variables")
        
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
                print("Using OpenAI client for DASHSCOPE API")
            except:
                self.use_openai_client = False
        except ImportError:
            print("OpenAI library not found, using requests instead")
            self.use_openai_client = False
        
        if not self.use_openai_client:
            print("Will use requests library for DASHSCOPE API calls")
        
        # Embedded metadata for video 000909
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
            
            # Create VideoMetadata from loaded data
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
            # Create default metadata if file doesn't exist
            print(f"Warning: metadata.json not found in {video_folder} and no embedded metadata available")
            print("Creating default metadata...")
            
            # Count frames in the folder
            frames = self.get_available_frames(video_folder)
            total_frames = len(frames) if frames else 0
            
            metadata = VideoMetadata(
                video_id=video_folder.name,
                weather=1,  # Default to sunny
                light=1,    # Default to day
                scene=4,    # Default to urban
                road=1,     # Default to arterials
                accident=False,
                abnormal_start=0,
                abnormal_end=0,
                accident_frame=0,
                total_frames=total_frames,
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
            print(f"Looking for frames in {images_dir}")
        else:
            search_dir = video_folder
            print(f"Looking for frames in {video_folder}")
        
        # Try different common frame naming patterns
        patterns = [
            "*.jpg", "*.jpeg", "*.png",
            "frame_*.jpg", "frame_*.png",
            "*_frame.jpg", "*_frame.png",
            "0*.jpg", "0*.png"  # For numbered frames like 000001.jpg
        ]
        
        for pattern in patterns:
            found_frames = list(search_dir.glob(pattern))
            if found_frames:
                frames.extend(found_frames)
                print(f"Found {len(found_frames)} frames with pattern {pattern}")
        
        # Remove duplicates and sort
        frames = list(set(frames))
        
        # Sort frames numerically based on the number in the filename
        def extract_frame_number(path):
            # Extract all numbers from filename
            numbers = re.findall(r'\d+', path.stem)
            if numbers:
                # Return the first number found (should be frame number)
                return int(numbers[0])
            return 0
        
        frames.sort(key=extract_frame_number)
        
        return frames
    
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
        
        # Build message content with multiple images
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
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=180  # Increased timeout for many images
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            raise Exception("API request timed out")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def select_representative_frames(self, video_metadata: VideoMetadata, 
                                    available_frames: List[Path], 
                                    max_frames: int = 15) -> List[Tuple[int, Path]]:
        """
        Select representative frames from the video for comprehensive understanding
        
        Returns:
            List of tuples (frame_number, frame_path)
        """
        total_frames = len(available_frames)
        
        if video_metadata.accident and video_metadata.abnormal_start > 0:
            # For accident videos, prioritize incident-related frames
            # Note: Frame numbers in metadata are 1-based, but list indices are 0-based
            key_frames = []
            
            # Calculate key frame numbers (1-based as in the metadata)
            key_moments = [
                1,  # Start of video
                10, # Normal driving
                18, # Just before abnormal (frame 19 is abnormal start)
                19, # Start of abnormal
                24, # Early abnormal phase
                29, # Mid pre-accident
                33, # Just before accident
                34, # Accident moment (car door opens hitting motorcycle)
                35, # Just after accident
                39, # Mid recovery
                44, # End of abnormal
                50, # After recovery
                63  # End of video
            ]
            
            # Map frame numbers to available frames
            selected_frames = []
            for frame_num in key_moments:
                # Find the corresponding frame file (frames are named 000001.jpg, 000002.jpg, etc.)
                for frame_path in available_frames:
                    # Extract frame number from filename
                    numbers = re.findall(r'\d+', frame_path.stem)
                    if numbers and int(numbers[0]) == frame_num:
                        selected_frames.append((frame_num, frame_path))
                        break
            
            # If we couldn't find specific frames, fall back to even distribution
            if len(selected_frames) < max_frames // 2:
                print("Warning: Could not find all key frames, using even distribution")
                step = max(1, total_frames // max_frames)
                selected_frames = []
                for i in range(0, total_frames, step):
                    if i < total_frames:
                        frame_path = available_frames[i]
                        # Extract frame number from filename
                        numbers = re.findall(r'\d+', frame_path.stem)
                        frame_num = int(numbers[0]) if numbers else i + 1
                        selected_frames.append((frame_num, frame_path))
                    if len(selected_frames) >= max_frames:
                        break
        else:
            # For non-accident videos or videos without metadata, sample evenly
            step = max(1, total_frames // max_frames)
            selected_frames = []
            for i in range(0, total_frames, step):
                if i < total_frames:
                    frame_path = available_frames[i]
                    # Extract frame number from filename
                    numbers = re.findall(r'\d+', frame_path.stem)
                    frame_num = int(numbers[0]) if numbers else i + 1
                    selected_frames.append((frame_num, frame_path))
                if len(selected_frames) >= max_frames:
                    break
        
        # Sort by frame number and limit to max_frames
        selected_frames = sorted(selected_frames, key=lambda x: x[0])[:max_frames]
        
        return selected_frames
    
    def create_comprehensive_grounding_prompt(self, 
                                             video_metadata: VideoMetadata,
                                             num_frames: int) -> str:
        """
        Create a prompt for generating a single grounded caption for the entire video with two sections
        """
        
        # Special handling for video 000909
        if video_metadata.video_id == "000909":
            incident_description = """This is a door-opening accident where a parked vehicle opens its door into the path of passing motorcycles.
The incident occurs at night on an urban arterial road. The vehicle does not check for approaching traffic before opening the car door,
resulting in a collision/near-miss with motorcycles passing by."""
        else:
            incident_description = video_metadata.causes if video_metadata.accident else "No incident"
        
        prompt = f"""You are analyzing a complete traffic video sequence. I'm showing you {num_frames} representative frames that capture the entire event from start to finish.

VIDEO INFORMATION:
- Video ID: {video_metadata.video_id}
- Total frames: {video_metadata.total_frames}
- Weather: {video_metadata.get_weather_str()}
- Time: {video_metadata.get_light_str()}
- Location: {video_metadata.get_scene_str()} {video_metadata.get_road_str()}
- Incident: {"Yes" if video_metadata.accident else "No"}

INCIDENT DETAILS:
{incident_description}

INCIDENT TIMELINE:
- Normal conditions: Frames 1-{video_metadata.abnormal_start-1}
- Abnormal conditions begin: Frame {video_metadata.abnormal_start}
- Accident occurs: Frame {video_metadata.accident_frame}
- Recovery period: Frames {video_metadata.accident_frame+1}-{video_metadata.abnormal_end}
- Return to normal: Frame {video_metadata.abnormal_end+1} onwards

SAFETY CONTEXT:
{video_metadata.measures}

YOUR TASK:
Generate a comprehensive grounded caption in TWO DISTINCT SECTIONS:

1. PRE-ACCIDENT SECTION (Frames 1-{video_metadata.accident_frame-1}): Use NORMAL tags
2. ACCIDENT AND RECOVERY SECTION (Frame {video_metadata.accident_frame} onwards): Use SPECIAL ACCIDENT tags

FORMATTING REQUIREMENTS:

SECTION 1 - PRE-ACCIDENT SECTION (Normal Tags):
----------------------------------------
Use standard XML tags for all objects before the accident:
- <road>, <lane>, <sidewalk>, <barrier>, <lane_marking>
- <car>, <truck>, <bus>, <motorcycle>, <vehicle>, <parked_car>
- <building>, <tree>, <sky>, <street_light>, <traffic_sign>, <billboard>
- <pedestrian>, <motorcyclist>, <driver>
- <headlight>, <taillight> (for night scenes)

Describe:
- The initial scene setup and environment
- Normal traffic flow and conditions
- Any developing abnormal conditions leading to the accident
- Continue until just before Frame {video_metadata.accident_frame}

SECTION 2 - ACCIDENT AND RECOVERY SECTION (Accident Tags):
----------------------------------------
Starting at Frame {video_metadata.accident_frame}, use SPECIAL ACCIDENT TAGS for all objects involved in or affected by the accident:
- <accident_object_1>primary accident object (e.g., the opening door)</accident_object_1>
- <accident_object_2>affected vehicle/motorcycle</accident_object_2>
- <accident_object_3>other affected vehicles</accident_object_3>
- <accident_object_4>debris or secondary objects</accident_object_4>
- Continue numbering as needed...

For objects NOT involved in the accident, continue using normal tags.

Describe:
- The exact moment of the accident at Frame {video_metadata.accident_frame}
- Immediate reactions and consequences
- Recovery period
- Return to normal conditions

EXAMPLE FORMAT:

PRE-ACCIDENT SECTION (Normal Tags):
----------------------------------------
The video begins on a <road>multi-lane urban arterial road</road> at night with <street_light>street lights illuminating the scene</street_light>. Several <motorcycle>motorcycles</motorcycle> travel along the <lane>right lane</lane> while a <parked_car>vehicle is parked</parked_car> on the roadside. The <headlight>motorcycle headlights</headlight> are clearly visible in the darkness. As the sequence progresses through Frames 1-{video_metadata.abnormal_start-1}, traffic flows normally with <motorcycle>motorcycles</motorcycle> maintaining steady speeds...

[Continue describing normal conditions and developing situation up to Frame {video_metadata.accident_frame-1}]

At **

ACCIDENT AND RECOVERY SECTION (Accident Tags):
----------------------------------------
Frame {video_metadata.accident_frame}**, the accident occurs: <accident_object_1>the parked vehicle's door suddenly opens</accident_object_1> directly into the path of <accident_object_2>an approaching motorcycle</accident_object_2>. The <accident_object_2>motorcycle</accident_object_2> swerves to avoid <accident_object_1>the opening door</accident_object_1>, while <accident_object_3>other motorcycles behind</accident_object_3> brake suddenly. The <accident_object_1>car door</accident_object_1> remains open as <accident_object_2>the affected motorcyclist</accident_object_2> manages to avoid direct collision...

[Continue describing the accident and recovery using accident_object tags for involved items]

Now generate the comprehensive grounded caption following this exact format:"""
        
        return prompt
    
    def parse_grounded_captions(self, response: str) -> Dict:
        """Parse the response to extract tagged objects and their descriptions"""
        # Find all XML-style tags including accident_object tags
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
            
            # Categorize objects
            if tag.startswith("accident_object"):
                accident_objects.append(obj)
            else:
                normal_objects.append(obj)
            
            # Count occurrences of each object type
            if tag not in object_counts:
                object_counts[tag] = 0
            object_counts[tag] += 1
        
        # Check if the response has the two-section format
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
    
    def generate_grounded_caption_for_video(self,
                                           video_folder: str,
                                           max_frames: int = 20,
                                           model: str = "qwen-vl-max",
                                           output_dir: str = "grounded_results") -> Dict:
        """
        Generate a single comprehensive grounded caption for the entire video
        
        Args:
            video_folder: Path to video folder containing frames and metadata
            max_frames: Maximum number of frames to send to API
            model: Model to use
            output_dir: Output directory for results
            
        Returns:
            Dictionary with the grounded caption and metadata
        """
        try:
            video_folder = Path(video_folder)
            if not video_folder.exists():
                raise ValueError(f"Video folder {video_folder} does not exist")
            
            video_id = video_folder.name
            print(f"\nGenerating comprehensive grounded caption for video {video_id}")
            
            # Load video metadata
            video_metadata = self.load_video_metadata(video_folder)
            print(f"Video contains {video_metadata.total_frames} total frames")
            
            # Print incident information for video 000909
            if video_id == "000909":
                print("\n" + "=" * 60)
                print("INCIDENT INFORMATION:")
                print(f"Type: Door-opening accident")
                print(f"Cause: {video_metadata.causes}")
                print(f"Time: Night, Urban arterial road")
                print(f"Critical frames: {video_metadata.abnormal_start}-{video_metadata.abnormal_end}")
                print(f"Accident frame: {video_metadata.accident_frame}")
                print("=" * 60)
            
            # Get all available frames
            available_frames = self.get_available_frames(video_folder)
            print(f"Found {len(available_frames)} frame files")
            
            if not available_frames:
                raise ValueError(f"No frame files found in {video_folder}")
            
            # Print first few frame names for debugging
            print(f"First few frames: {[f.name for f in available_frames[:5]]}")
            
            # Select representative frames
            selected_frames = self.select_representative_frames(
                video_metadata, 
                available_frames, 
                max_frames
            )
            
            print(f"Selected {len(selected_frames)} representative frames")
            frame_numbers = [num for num, _ in selected_frames]
            print(f"  Frame numbers: {frame_numbers}")
            
            # Prepare frames for API call
            frames_for_api = []
            
            # Add introduction text
            frames_for_api.append({
                "text": f"I'm showing you {len(selected_frames)} frames from a {video_metadata.total_frames}-frame traffic video. "
                       f"These frames are selected to show the complete sequence of events.\n\n"
                       f"Frame numbers being shown: {frame_numbers}\n"
            })
            
            # Add frames with descriptions
            for frame_num, frame_path in selected_frames:
                # Determine phase and section
                if frame_num < video_metadata.accident_frame:
                    section = "PRE-ACCIDENT"
                    if frame_num < video_metadata.abnormal_start:
                        phase = "Normal driving"
                    else:
                        phase = "Abnormal developing"
                else:
                    section = "ACCIDENT/RECOVERY"
                    if frame_num == video_metadata.accident_frame:
                        phase = "ACCIDENT MOMENT"
                    elif frame_num <= video_metadata.abnormal_end:
                        phase = "Recovery"
                    else:
                        phase = "Return to normal"
                
                # Add frame description
                frames_for_api.append({
                    "text": f"\nFrame {frame_num} ({section} - {phase}):"
                })
                
                # Add the image
                frames_for_api.append({
                    "image_path": str(frame_path)
                })
            
            # Add the main prompt
            prompt = self.create_comprehensive_grounding_prompt(
                video_metadata,
                len(selected_frames)
            )
            
            frames_for_api.append({
                "text": prompt
            })
            
            # Call API with all frames
            print(f"\nCalling DASHSCOPE API with {len(selected_frames)} frames...")
            print("This may take a moment due to multiple images...")
            
            if self.use_openai_client:
                # Build content for OpenAI client
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
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=4000,
                    temperature=0.1
                )
                response = completion.choices[0].message.content
            else:
                response = self.call_api_with_requests_multi_image(
                    frames_for_api,
                    model=model,
                    max_tokens=4000,
                    temperature=0.1
                )
            
            print("API call completed successfully")
            
            # Parse the response
            grounded_output = self.parse_grounded_captions(response)
            
            # Prepare comprehensive result
            result = {
                "video_id": video_id,
                "video_metadata": {
                    "total_frames": video_metadata.total_frames,
                    "weather": video_metadata.get_weather_str(),
                    "light": video_metadata.get_light_str(),
                    "scene": video_metadata.get_scene_str(),
                    "road": video_metadata.get_road_str(),
                    "accident": video_metadata.accident,
                    "accident_frame": video_metadata.accident_frame,
                    "causes": video_metadata.causes if video_metadata.accident else None,
                    "measures": video_metadata.measures if video_metadata.accident else None
                },
                "processing_info": {
                    "total_available_frames": len(available_frames),
                    "frames_used": len(selected_frames),
                    "frame_numbers_used": frame_numbers
                },
                "grounded_caption": grounded_output,
                "raw_response": response
            }
            
            # Save results
            output_path = Path(output_dir) / video_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save comprehensive result
            output_file = output_path / f"{video_id}_comprehensive_grounded_caption.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved comprehensive result to: {output_file}")
            
            # Also save just the caption in a text file for easy reading
            caption_file = output_path / f"{video_id}_grounded_caption.txt"
            with open(caption_file, 'w') as f:
                f.write("COMPREHENSIVE GROUNDED CAPTION\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Video: {video_id}\n")
                f.write(f"Type: Door-opening accident\n" if video_id == "000909" else "")
                f.write(f"Conditions: {video_metadata.get_light_str()}, {video_metadata.get_weather_str()}\n")
                f.write(f"Location: {video_metadata.get_scene_str()} {video_metadata.get_road_str()}\n")
                f.write(f"Accident Frame: {video_metadata.accident_frame}\n\n")
                f.write("CAPTION:\n")
                f.write("-" * 60 + "\n")
                f.write(result['grounded_caption']['full_caption'])
                f.write("\n" + "-" * 60 + "\n\n")
                f.write(f"Total grounded objects: {len(result['grounded_caption']['objects'])}\n")
                f.write(f"Normal objects: {result['grounded_caption']['total_normal_objects']}\n")
                f.write(f"Accident objects: {result['grounded_caption']['total_accident_objects']}\n")
                f.write(f"Unique object types: {', '.join(result['grounded_caption']['unique_object_types'])}\n")
                f.write(f"Has two sections: {result['grounded_caption']['has_two_sections']}\n")
            print(f"Saved caption text to: {caption_file}")
            
            # Print summary
            print("\n" + "=" * 60)
            print("COMPREHENSIVE GROUNDED CAPTION GENERATED")
            print("=" * 60)
            
            # Check if the format is correct
            if grounded_output['has_two_sections']:
                print("✓ Caption correctly formatted with two sections")
            else:
                print("⚠ Warning: Caption may not have the correct two-section format")
            
            print("\nCaption Structure:")
            print("-" * 60)
            if "PRE-ACCIDENT SECTION" in response:
                print("✓ PRE-ACCIDENT SECTION found")
            if "ACCIDENT AND RECOVERY SECTION" in response:
                print("✓ ACCIDENT AND RECOVERY SECTION found")
            
            print(f"\nStatistics:")
            print(f"  - Frames analyzed: {result['processing_info']['frames_used']}/{result['processing_info']['total_available_frames']}")
            print(f"  - Total grounded objects: {len(result['grounded_caption']['objects'])}")
            print(f"  - Normal tagged objects: {result['grounded_caption']['total_normal_objects']}")
            print(f"  - Accident tagged objects: {result['grounded_caption']['total_accident_objects']}")
            print(f"  - Unique object types: {len(result['grounded_caption']['unique_object_types'])}")
            
            # Object frequency
            if result['grounded_caption']['object_counts']:
                print(f"\nObject frequency in caption:")
                
                # Separate normal and accident objects
                normal_counts = {k: v for k, v in result['grounded_caption']['object_counts'].items() 
                               if not k.startswith('accident_object')}
                accident_counts = {k: v for k, v in result['grounded_caption']['object_counts'].items() 
                                 if k.startswith('accident_object')}
                
                if normal_counts:
                    print("\n  Normal objects:")
                    for obj_type, count in sorted(normal_counts.items()):
                        print(f"    - {obj_type}: {count} occurrences")
                
                if accident_counts:
                    print("\n  Accident objects:")
                    for obj_type, count in sorted(accident_counts.items()):
                        print(f"    - {obj_type}: {count} occurrences")
            
            print("\n" + "-" * 60)
            print("Caption Preview (first 500 chars):")
            print(response[:500] + "...")
            
            return result
            
        except Exception as e:
            print(f"Error generating grounded caption: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to generate a single comprehensive grounded caption for the entire video"""
    
    # Set up API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable")
        api_key = input("Or enter your DASHSCOPE API key: ").strip()
    
    # Initialize the generator
    generator = GroundedCaptionGenerator(api_key=api_key)
    
    # Path to video folder - using 000909 as specified
    video_folder = "000909"  # This is the door-opening accident video
    
    # Generate single comprehensive grounded caption for entire video
    print("\n" + "=" * 70)
    print(f"GENERATING COMPREHENSIVE GROUNDED CAPTION FOR VIDEO 000909")
    print("Type: Door-opening accident at night on urban road")
    print("Format: Two-section caption with normal and accident tags")
    print("=" * 70)
    
    try:
        result = generator.generate_grounded_caption_for_video(
            video_folder=video_folder,
            max_frames=20,  # Use up to 20 frames for comprehensive understanding
            model="qwen-vl-max",
            output_dir="grounded_results"
        )
        
        if result:
            print(f"\n✓ Successfully generated comprehensive grounded caption")
            print(f"✓ Used {result['processing_info']['frames_used']} frames from the video")
            print(f"✓ Grounded {len(result['grounded_caption']['objects'])} object instances")
            print(f"✓ Normal objects: {result['grounded_caption']['total_normal_objects']}")
            print(f"✓ Accident objects: {result['grounded_caption']['total_accident_objects']}")
            print(f"\n✓ Key incident: Door-opening collision with motorcycle at frame {34}")
        else:
            print("\n✗ Failed to generate comprehensive grounded caption")
            
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()