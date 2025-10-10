import cv2
import numpy as np
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Dict, Tuple, Optional, Union, Set
import re
import os
from pathlib import Path
from dataclasses import dataclass
from glob import glob

@dataclass
class MaskObject:
    """Data class for storing mask object information"""
    object_id: int
    bbox: List[int]
    mask_file: str
    score: float
    object_label: str
    type: str
    description: str
    mask_area: int
    mask_array: Optional[np.ndarray] = None

@dataclass
class VideoMetadata:
    """Data class for video metadata from dense captions JSON"""
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
    
    def load_dense_captions_json(self, json_path: str) -> Dict[str, VideoMetadata]:
        """Load dense captions from the provided JSON format"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        video_data = {}
        for item in data:
            metadata = VideoMetadata(
                video_id=item["Video"],
                weather=item["weather(sunny,rainy,snowy,foggy)1-4"],
                light=item["light(day,night)1-2"],
                scene=item["scenes(highway,tunnel,mountain,urban,rural)1-5"],
                road=item["linear(arterials,curve,intersection,T-junction,ramp) 1-5"],
                accident=bool(item["whether an accident occurred (1/0)"]),
                abnormal_start=item["abnormal start frame"],
                abnormal_end=item["abnormal end frame"],
                accident_frame=item["accident frame"],
                total_frames=item["total frames"],
                dense_caption=item["dense_caption"],
                causes=item.get("causes", ""),
                measures=item.get("measures", "")
            )
            video_data[metadata.video_id] = metadata
            
        return video_data
    
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
                timeout=180
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
                                    available_frames: List[int], 
                                    max_frames: int = 15) -> List[int]:
        """
        Select representative frames from the video for comprehensive understanding
        """
        selected = []
        available_frames = sorted(available_frames)
        
        if video_metadata.accident:
            # For accident videos, prioritize incident-related frames
            critical_frames = []
            
            # Always include these if available
            key_moments = [
                0,  # Start of video
                max(0, video_metadata.abnormal_start - 3),  # Just before abnormal
                video_metadata.abnormal_start,  # Start of abnormal
                (video_metadata.abnormal_start + video_metadata.accident_frame) // 2,  # Mid pre-accident
                video_metadata.accident_frame - 1,  # Just before accident
                video_metadata.accident_frame,  # Accident moment
                video_metadata.accident_frame + 1,  # Just after accident
                (video_metadata.accident_frame + video_metadata.abnormal_end) // 2,  # Mid recovery
                video_metadata.abnormal_end,  # End of abnormal
                min(video_metadata.total_frames - 1, video_metadata.abnormal_end + 5),  # After recovery
                video_metadata.total_frames - 1  # End of video
            ]
            
            for moment in key_moments:
                if moment in available_frames and moment not in critical_frames:
                    critical_frames.append(moment)
            
            selected = critical_frames[:max_frames]
            
            # If we have room for more frames, add some evenly distributed
            if len(selected) < max_frames:
                remaining_frames = [f for f in available_frames if f not in selected]
                step = max(1, len(remaining_frames) // (max_frames - len(selected)))
                for i in range(0, len(remaining_frames), step):
                    if len(selected) >= max_frames:
                        break
                    selected.append(remaining_frames[i])
        else:
            # For non-accident videos, sample evenly
            step = max(1, len(available_frames) // max_frames)
            selected = available_frames[::step][:max_frames]
        
        return sorted(selected)
    
    def collect_all_objects_from_video(self, video_id: str, 
                                      base_path: str = "segmentation_output") -> Dict[str, Set[str]]:
        """
        Collect all unique objects that appear throughout the video
        """
        metadata_dir = Path(base_path) / video_id / "metadata"
        all_objects = {}
        
        for metadata_file in metadata_dir.glob("*_masks.json"):
            with open(metadata_file, 'r') as f:
                frame_data = json.load(f)
            
            for obj in frame_data.get('objects', []):
                obj_type = obj.get('type', 'unknown')
                if obj_type not in all_objects:
                    all_objects[obj_type] = set()
                
                description = obj.get('description', '')
                if description:
                    all_objects[obj_type].add(description)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in all_objects.items()}
    
    def create_comprehensive_grounding_prompt_with_accident_tags(self, 
                                                                video_metadata: VideoMetadata,
                                                                all_objects: Dict[str, List[str]],
                                                                num_frames: int) -> str:
        """
        Create a prompt for generating a single grounded caption with special accident object tagging
        """
        # Format object information
        object_summary = []
        for obj_type, descriptions in sorted(all_objects.items()):
            count = len(descriptions)
            if descriptions:
                sample_descs = descriptions[:3]
                obj_text = f"- {obj_type}: {count} instances ({', '.join(sample_descs)}...)" if count > 3 else f"- {obj_type}: {', '.join(descriptions)}"
            else:
                obj_text = f"- {obj_type}: appears in video"
            object_summary.append(obj_text)
        
        objects_text = "\n".join(object_summary)
        
        # Determine if this is an accident video
        accident_section = ""
        if video_metadata.accident:
            accident_section = f"""
CRITICAL ACCIDENT INFORMATION:
- Accident occurs at Frame {video_metadata.accident_frame}
- Cause: {video_metadata.causes}
- Abnormal period: Frames {video_metadata.abnormal_start} to {video_metadata.abnormal_end}

SPECIAL TAGGING RULES FOR ACCIDENT:
Starting from Frame {video_metadata.accident_frame} (the accident moment), you MUST:
1. Use special tags <accident_object_N> for objects directly involved in or affected by the accident
2. Number accident objects sequentially: <accident_object_1>, <accident_object_2>, etc.
3. Keep the same accident object number consistent throughout (e.g., the tire that falls off is always accident_object_1)
4. Objects not involved in the accident continue using normal tags

ACCIDENT OBJECT EXAMPLES:
- <accident_object_1>tire that detaches</accident_object_1>
- <accident_object_2>vehicle that loses the tire</accident_object_2>
- <accident_object_3>vehicles that swerve to avoid</accident_object_3>
- <accident_object_4>debris on road</accident_object_4>

The transition should be clear: normal tags before Frame {video_metadata.accident_frame}, then accident tags for involved objects after."""
        
        prompt = f"""You are analyzing a complete traffic video sequence. I'm showing you {num_frames} representative frames that capture the entire event from start to finish.

VIDEO INFORMATION:
- Video ID: {video_metadata.video_id}
- Total frames: {video_metadata.total_frames}
- Weather: {video_metadata.get_weather_str()}
- Time: {video_metadata.get_light_str()}
- Location: {video_metadata.get_scene_str()} {video_metadata.get_road_str()}
{accident_section}

OBJECTS DETECTED THROUGHOUT THE VIDEO:
{objects_text}

FULL VIDEO DESCRIPTION:
{video_metadata.dense_caption}

YOUR TASK:
Generate a SINGLE comprehensive grounded caption that describes the ENTIRE video sequence from beginning to end.

REQUIREMENTS:
1. Create ONE flowing narrative that covers the complete video chronologically
2. For frames BEFORE the accident (Frames 0-{video_metadata.accident_frame-1 if video_metadata.accident else video_metadata.total_frames}):
   - Use normal XML tags: <object_type>description</object_type>
   - Example: <car>white sedan</car>, <road>multi-lane highway</road>
   
3. For frames DURING AND AFTER the accident (Frame {video_metadata.accident_frame if video_metadata.accident else 'N/A'} onwards):
   - Use <accident_object_N> tags for objects involved in the accident
   - Continue using normal tags for uninvolved objects
   - Maintain consistent numbering for each accident object
   
4. Structure your caption with clear progression:
   - Normal driving conditions (normal tags)
   - Pre-accident changes if any (normal tags)
   - ACCIDENT MOMENT at Frame {video_metadata.accident_frame if video_metadata.accident else 'N/A'} (introduce accident tags)
   - Recovery phase (maintain accident tags for affected objects)
   - Return to normal (mix of tags as appropriate)

EXAMPLE FORMAT:
"The video begins on a <road>multi-lane urban road</road> where <car>several vehicles</car> travel under <sky>clear skies</sky>. <tree>Trees</tree> line the <sidewalk>sidewalk</sidewalk>... [Normal driving continues]... 

At Frame {video_metadata.accident_frame if video_metadata.accident else 'XX'}, the accident occurs: <accident_object_1>a tire detaches from the sedan</accident_object_1>, causing <accident_object_2>the affected vehicle</accident_object_2> to lose control. <accident_object_3>Nearby vehicles</accident_object_3> swerve to avoid <accident_object_1>the rolling tire</accident_object_1>. The <road>road surface</road> now has <accident_object_4>scattered debris</accident_object_4>... [Recovery continues with accident tags]..."

Now generate the comprehensive grounded caption with appropriate tagging:"""
        
        return prompt
    
    def parse_grounded_captions_with_accident_tags(self, response: str, accident_frame: Optional[int] = None) -> Dict:
        """
        Parse the response to extract tagged objects, distinguishing between normal and accident tags
        """
        # Find all XML-style tags (both normal and accident)
        normal_pattern = r'<([^/\s>]+?)>(.*?)</\1>'
        matches = re.findall(normal_pattern, response, re.DOTALL)
        
        grounded_objects = []
        normal_objects = []
        accident_objects = []
        object_counts = {}
        accident_object_mapping = {}
        
        for tag, description in matches:
            obj_entry = {
                "object_type": tag,
                "description": description.strip()
            }
            
            # Check if it's an accident object
            if tag.startswith('accident_object_'):
                accident_objects.append(obj_entry)
                # Extract accident object number
                match = re.match(r'accident_object_(\d+)', tag)
                if match:
                    obj_num = int(match.group(1))
                    if obj_num not in accident_object_mapping:
                        accident_object_mapping[obj_num] = []
                    accident_object_mapping[obj_num].append(description.strip())
            else:
                normal_objects.append(obj_entry)
            
            grounded_objects.append(obj_entry)
            
            # Count occurrences
            if tag not in object_counts:
                object_counts[tag] = 0
            object_counts[tag] += 1
        
        # Identify where accident description starts in the caption
        accident_section_start = None
        if accident_frame is not None:
            # Look for mention of the accident frame
            frame_mention = re.search(f'Frame {accident_frame}', response)
            if frame_mention:
                accident_section_start = frame_mention.start()
        
        return {
            "full_caption": response,
            "all_objects": grounded_objects,
            "normal_objects": normal_objects,
            "accident_objects": accident_objects,
            "accident_object_mapping": accident_object_mapping,
            "object_counts": object_counts,
            "unique_object_types": list(set([obj["object_type"] for obj in normal_objects])),
            "unique_accident_objects": list(accident_object_mapping.keys()),
            "accident_section_start_pos": accident_section_start,
            "total_normal_tags": len(normal_objects),
            "total_accident_tags": len(accident_objects)
        }
    
    def generate_single_grounded_caption_for_video(self,
                                                  video_id: str,
                                                  video_metadata: VideoMetadata,
                                                  base_path: str = "segmentation_output",
                                                  max_frames: int = 20,
                                                  model: str = "qwen-vl-max") -> Dict:
        """
        Generate a single comprehensive grounded caption with accident object tagging
        """
        try:
            print(f"\nGenerating comprehensive grounded caption for video {video_id}")
            print(f"Video contains {video_metadata.total_frames} total frames")
            if video_metadata.accident:
                print(f"ACCIDENT DETECTED at frame {video_metadata.accident_frame}")
                print(f"Will use special accident object tagging from frame {video_metadata.accident_frame} onwards")
            
            # Get all available overlay frames
            overlays_dir = Path(base_path) / video_id / "overlays"
            available_overlays = sorted(overlays_dir.glob("*_overlay.jpg"))
            
            # Extract frame numbers
            available_frame_numbers = []
            for overlay in available_overlays:
                match = re.search(r'(\d+)_overlay\.jpg', overlay.name)
                if match:
                    available_frame_numbers.append(int(match.group(1)))
            
            print(f"Found {len(available_frame_numbers)} overlay frames")
            
            # Select representative frames
            selected_frame_numbers = self.select_representative_frames(
                video_metadata, 
                available_frame_numbers, 
                max_frames
            )
            
            print(f"Selected {len(selected_frame_numbers)} representative frames:")
            print(f"  Frames: {selected_frame_numbers}")
            
            # Collect all objects from the entire video
            print("\nCollecting all objects from video...")
            all_objects = self.collect_all_objects_from_video(video_id, base_path)
            print(f"Found {len(all_objects)} unique object types across all frames")
            
            # Prepare frames for API call
            frames_for_api = []
            
            # Add introduction text
            intro_text = f"I'm showing you {len(selected_frame_numbers)} frames from a {video_metadata.total_frames}-frame traffic video."
            if video_metadata.accident:
                intro_text += f" An accident occurs at Frame {video_metadata.accident_frame}. Pay special attention to the transition from normal to accident conditions."
            intro_text += f"\n\nFrame sequence: {selected_frame_numbers}\n"
            
            frames_for_api.append({"text": intro_text})
            
            # Add frames with descriptions
            for frame_num in selected_frame_numbers:
                frame_str = str(frame_num).zfill(6)
                overlay_path = overlays_dir / f"{frame_str}_overlay.jpg"
                
                if overlay_path.exists():
                    # Determine phase and special marking for accident frame
                    phase_text = ""
                    if video_metadata.accident:
                        if frame_num < video_metadata.abnormal_start:
                            phase_text = "Normal driving"
                        elif frame_num < video_metadata.accident_frame:
                            phase_text = "Pre-accident (conditions changing)"
                        elif frame_num == video_metadata.accident_frame:
                            phase_text = "*** ACCIDENT OCCURS HERE - START USING ACCIDENT TAGS ***"
                        elif frame_num <= video_metadata.abnormal_end:
                            phase_text = "Post-accident (use accident tags for affected objects)"
                        else:
                            phase_text = "Return to normal"
                    else:
                        phase_text = "Normal driving"
                    
                    # Add frame description
                    frames_for_api.append({
                        "text": f"\nFrame {frame_num} ({phase_text}):"
                    })
                    
                    # Add the image
                    frames_for_api.append({
                        "image_path": str(overlay_path)
                    })
            
            # Add the main prompt with accident tagging instructions
            prompt = self.create_comprehensive_grounding_prompt_with_accident_tags(
                video_metadata,
                all_objects,
                len(selected_frame_numbers)
            )
            
            frames_for_api.append({"text": prompt})
            
            # Call API with all frames
            print(f"\nCalling DASHSCOPE API with {len(selected_frame_numbers)} frames...")
            print("Generating caption with accident object tagging...")
            
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
            
            # Parse the response with accident tag awareness
            grounded_output = self.parse_grounded_captions_with_accident_tags(
                response, 
                video_metadata.accident_frame if video_metadata.accident else None
            )
            
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
                    "accident_frame": video_metadata.accident_frame if video_metadata.accident else None,
                    "causes": video_metadata.causes if video_metadata.accident else None,
                    "measures": video_metadata.measures if video_metadata.accident else None
                },
                "processing_info": {
                    "total_available_frames": len(available_frame_numbers),
                    "frames_used": len(selected_frame_numbers),
                    "frame_numbers_used": selected_frame_numbers,
                    "all_object_types_in_video": list(all_objects.keys()),
                    "total_unique_objects": len(all_objects)
                },
                "grounded_caption": grounded_output,
                "tagging_statistics": {
                    "normal_objects_tagged": grounded_output["total_normal_tags"],
                    "accident_objects_tagged": grounded_output["total_accident_tags"],
                    "unique_accident_objects": len(grounded_output["unique_accident_objects"]),
                    "accident_object_descriptions": grounded_output["accident_object_mapping"]
                },
                "raw_response": response
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating grounded caption: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_video_comprehensive(self,
                                   dense_captions_json: str,
                                   video_id: str,
                                   base_path: str = "segmentation_output",
                                   output_dir: str = "grounded_results",
                                   max_frames: int = 20,
                                   model: str = "qwen-vl-max") -> Dict:
        """
        Process entire video to generate a single comprehensive grounded caption with accident tagging
        """
        # Load video metadata
        print(f"Loading dense captions from {dense_captions_json}")
        video_metadata_dict = self.load_dense_captions_json(dense_captions_json)
        
        if video_id not in video_metadata_dict:
            raise ValueError(f"Video {video_id} not found in dense captions JSON")
        
        video_metadata = video_metadata_dict[video_id]
        
        print("\n" + "=" * 70)
        print(f"PROCESSING VIDEO: {video_id}")
        print("=" * 70)
        print(f"Scene: {video_metadata.get_scene_str()} {video_metadata.get_road_str()}")
        print(f"Conditions: {video_metadata.get_weather_str()}, {video_metadata.get_light_str()}")
        print(f"Total frames: {video_metadata.total_frames}")
        
        if video_metadata.accident:
            print(f"\n" + "!" * 70)
            print(f"ACCIDENT DETECTED")
            print(f"  - Type: {video_metadata.causes}")
            print(f"  - Accident frame: {video_metadata.accident_frame}")
            print(f"  - Abnormal period: frames {video_metadata.abnormal_start}-{video_metadata.abnormal_end}")
            print(f"  - Will use special accident object tagging")
            print("!" * 70)
        
        # Generate comprehensive grounded caption
        result = self.generate_single_grounded_caption_for_video(
            video_id=video_id,
            video_metadata=video_metadata,
            base_path=base_path,
            max_frames=max_frames,
            model=model
        )
        
        if result:
            # Create output directory
            output_path = Path(output_dir) / video_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save comprehensive result
            output_file = output_path / f"{video_id}_grounded_caption_with_accident_tags.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved comprehensive result to: {output_file}")
            
            # Save formatted caption for easy reading
            caption_file = output_path / f"{video_id}_grounded_caption_formatted.txt"
            with open(caption_file, 'w') as f:
                f.write("COMPREHENSIVE GROUNDED CAPTION WITH ACCIDENT TAGGING\n")
                f.write("=" * 70 + "\n\n")
                
                # Split caption to highlight accident section
                caption = result['grounded_caption']['full_caption']
                
                if video_metadata.accident and f"Frame {video_metadata.accident_frame}" in caption:
                    # Find where accident description starts
                    parts = caption.split(f"Frame {video_metadata.accident_frame}")
                    
                    f.write("PRE-ACCIDENT SECTION (Normal Tags):\n")
                    f.write("-" * 40 + "\n")
                    f.write(parts[0])
                    f.write("\n\n")
                    
                    f.write("ACCIDENT AND RECOVERY SECTION (Accident Tags):\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Frame {video_metadata.accident_frame}" + f"Frame {video_metadata.accident_frame}".join(parts[1:]))
                else:
                    f.write(caption)
                
                f.write("\n\n" + "=" * 70 + "\n")
                f.write("TAGGING STATISTICS:\n")
                f.write(f"  - Normal objects tagged: {result['tagging_statistics']['normal_objects_tagged']}\n")
                f.write(f"  - Accident objects tagged: {result['tagging_statistics']['accident_objects_tagged']}\n")
                f.write(f"  - Unique accident objects: {result['tagging_statistics']['unique_accident_objects']}\n")
                
                if result['tagging_statistics']['accident_object_descriptions']:
                    f.write("\nACCIDENT OBJECT MAPPING:\n")
                    for obj_num, descriptions in result['tagging_statistics']['accident_object_descriptions'].items():
                        f.write(f"  - accident_object_{obj_num}: {', '.join(descriptions)}\n")
            
            print(f"Saved formatted caption to: {caption_file}")
            
            # Print summary
            print("\n" + "=" * 70)
            print("CAPTION GENERATED WITH ACCIDENT TAGGING")
            print("=" * 70)
            
            # Show snippet of the caption focusing on accident if present
            if video_metadata.accident:
                caption = result['grounded_caption']['full_caption']
                accident_match = re.search(f'(Frame {video_metadata.accident_frame}.*?)(?:Frame \\d+|$)', caption, re.DOTALL)
                if accident_match:
                    print("\nACCIDENT SECTION EXCERPT:")
                    print("-" * 40)
                    print(accident_match.group(1)[:500] + "...")
            
            print(f"\nStatistics:")
            print(f"  - Frames analyzed: {result['processing_info']['frames_used']}/{result['processing_info']['total_available_frames']}")
            print(f"  - Normal objects tagged: {result['tagging_statistics']['normal_objects_tagged']}")
            print(f"  - Accident objects tagged: {result['tagging_statistics']['accident_objects_tagged']}")
            print(f"  - Unique accident objects: {result['tagging_statistics']['unique_accident_objects']}")
            
            if result['tagging_statistics']['accident_object_descriptions']:
                print(f"\nAccident Object Mapping:")
                for obj_num, descriptions in sorted(result['tagging_statistics']['accident_object_descriptions'].items()):
                    print(f"  - accident_object_{obj_num}: {descriptions[0] if descriptions else 'N/A'}")
            
            return result
        else:
            print("\nFailed to generate comprehensive grounded caption")
            return None


def main():
    """Main function to generate a comprehensive grounded caption with accident tagging"""
    
    # Set up API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable")
        api_key = input("Or enter your DASHSCOPE API key: ").strip()
    
    # Initialize the generator
    generator = GroundedCaptionGenerator(api_key=api_key)
    
    # Path to dense captions JSON
    dense_captions_json = "dense_captions_output_1.json"
    
    # Video ID to process
    video_id = "002320"
    
    # Generate comprehensive grounded caption with accident tagging
    print("\n" + "=" * 70)
    print(f"GENERATING GROUNDED CAPTION WITH ACCIDENT TAGGING")
    print(f"Video: {video_id}")
    print("=" * 70)
    
    try:
        result = generator.process_video_comprehensive(
            dense_captions_json=dense_captions_json,
            video_id=video_id,
            base_path="segmentation_output",
            output_dir="grounded_results",
            max_frames=20,  # Use up to 20 frames for comprehensive understanding
            model="qwen-vl-max"
        )
        
        if result:
            print(f"\n✓ Successfully generated comprehensive grounded caption with accident tagging")
            print(f"✓ Used {result['processing_info']['frames_used']} frames from the video")
            print(f"✓ Tagged {result['tagging_statistics']['normal_objects_tagged']} normal objects")
            print(f"✓ Tagged {result['tagging_statistics']['accident_objects_tagged']} accident objects")
        else:
            print("\n✗ Failed to generate comprehensive grounded caption")
            
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()