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
        mime_type = "image/jpeg" if path.suffix.lower() in ['', '.jpeg'] else "image/png"
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
                                    available_frames: List[int], 
                                    max_frames: int = 15) -> List[int]:
        """
        Select representative frames from the video for comprehensive understanding
        
        Args:
            video_metadata: Video metadata
            available_frames: List of available frame numbers
            max_frames: Maximum number of frames to select
            
        Returns:
            List of selected frame numbers
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
        
        Returns:
            Dictionary mapping object types to their descriptions
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
    
    def create_comprehensive_grounding_prompt(self, 
                                             video_metadata: VideoMetadata,
                                             all_objects: Dict[str, List[str]],
                                             num_frames: int) -> str:
        """
        Create a prompt for generating a single grounded caption for the entire video
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
        
        prompt = f"""You are analyzing a complete traffic video sequence. I'm showing you {num_frames} representative frames that capture the entire event from start to finish.

VIDEO INFORMATION:
- Video ID: {video_metadata.video_id}
- Total frames: {video_metadata.total_frames}
- Weather: {video_metadata.get_weather_str()}
- Time: {video_metadata.get_light_str()}
- Location: {video_metadata.get_scene_str()} {video_metadata.get_road_str()}
- Incident: {"Yes - " + video_metadata.causes if video_metadata.accident else "No incident"}

INCIDENT TIMELINE (if applicable):
- Normal driving: Frames 0-{video_metadata.abnormal_start-1 if video_metadata.accident else video_metadata.total_frames}
{f"- Abnormal conditions start: Frame {video_metadata.abnormal_start}" if video_metadata.accident else ""}
{f"- Accident occurs: Frame {video_metadata.accident_frame}" if video_metadata.accident else ""}
{f"- Recovery period: Frames {video_metadata.accident_frame+1}-{video_metadata.abnormal_end}" if video_metadata.accident else ""}
{f"- Return to normal: Frame {video_metadata.abnormal_end+1} onwards" if video_metadata.accident else ""}

OBJECTS DETECTED THROUGHOUT THE VIDEO:
{objects_text}

FULL VIDEO DESCRIPTION:
{video_metadata.dense_caption}

YOUR TASK:
Generate a SINGLE comprehensive grounded caption that describes the ENTIRE video sequence from beginning to end. 

REQUIREMENTS:
1. Create ONE flowing narrative that covers the complete video
2. Use XML-style tags <object_type>description</object_type> for EVERY type of object mentioned
3. Describe the progression of events chronologically
4. Include all major objects that appear in the video
5. For the incident (if present), describe:
   - The conditions before
   - The incident itself ({video_metadata.causes if video_metadata.accident else "N/A"})
   - The immediate aftermath
   - The recovery
6. Maintain narrative flow while grounding all objects

EXAMPLE FORMAT:
"The video begins on a <road>multi-lane urban arterial road</road> where <car>several vehicles</car> are traveling under <sky>clear sunny skies</sky>. <traffic_sign>Large billboards</traffic_sign> and <street_light>street lights</street_light> line the road while <tree>trees</tree> provide shade along <sidewalk>the sidewalk</sidewalk>. As the sequence progresses, <car>a white sedan</car> experiences a critical failure when its <debris>tire detaches</debris>, creating an immediate hazard. Other <car>vehicles</car> react by swerving to avoid the <debris>rolling tire</debris>..."

Now generate a comprehensive grounded caption for the ENTIRE video:"""
        
        return prompt
    
    def parse_grounded_captions(self, response: str) -> Dict:
        """Parse the response to extract tagged objects and their descriptions"""
        # Find all XML-style tags
        pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        grounded_objects = []
        object_counts = {}
        
        for tag, description in matches:
            grounded_objects.append({
                "object_type": tag,
                "description": description.strip()
            })
            
            # Count occurrences of each object type
            if tag not in object_counts:
                object_counts[tag] = 0
            object_counts[tag] += 1
        
        return {
            "full_caption": response,
            "objects": grounded_objects,
            "object_counts": object_counts,
            "unique_object_types": list(object_counts.keys())
        }
    
    def generate_single_grounded_caption_for_video(self,
                                                  video_id: str,
                                                  video_metadata: VideoMetadata,
                                                  base_path: str = "segmentation_output",
                                                  max_frames: int = 20,
                                                  model: str = "qwen-vl-max") -> Dict:
        """
        Generate a single comprehensive grounded caption for the entire video
        
        Args:
            video_id: Video identifier
            video_metadata: Video metadata
            base_path: Base path for segmentation data
            max_frames: Maximum number of frames to send to API
            model: Model to use
            
        Returns:
            Dictionary with the grounded caption and metadata
        """
        try:
            print(f"\nGenerating comprehensive grounded caption for video {video_id}")
            print(f"Video contains {video_metadata.total_frames} total frames")
            
            # Get all available overlay frames
            overlays_dir = Path(base_path) / video_id / "overlays"
            available_overlays = sorted(overlays_dir.glob("*_overlay"))
            
            # Extract frame numbers
            available_frame_numbers = []
            for overlay in available_overlays:
                match = re.search(r'(\d+)_overlay\', overlay.name)
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
            for obj_type, descriptions in all_objects.items():
                print(f"  - {obj_type}: {len(descriptions)} variations")
            
            # Prepare frames for API call
            frames_for_api = []
            
            # Add introduction text
            frames_for_api.append({
                "text": f"I'm showing you {len(selected_frame_numbers)} frames from a {video_metadata.total_frames}-frame traffic video. "
                       f"These frames are selected to show the complete sequence of events.\n\n"
                       f"Frame sequence being shown: {selected_frame_numbers}\n"
            })
            
            # Add frames with descriptions
            for frame_num in selected_frame_numbers:
                frame_str = str(frame_num).zfill(6)
                overlay_path = overlays_dir / f"{frame_str}_overlay"
                
                if overlay_path.exists():
                    # Determine phase
                    if frame_num < video_metadata.abnormal_start:
                        phase = "Normal driving"
                    elif frame_num < video_metadata.accident_frame:
                        phase = "Pre-accident (abnormal developing)"
                    elif frame_num == video_metadata.accident_frame:
                        phase = "ACCIDENT MOMENT"
                    elif frame_num <= video_metadata.abnormal_end:
                        phase = "Post-accident recovery"
                    else:
                        phase = "Return to normal"
                    
                    # Add frame description
                    frames_for_api.append({
                        "text": f"\nFrame {frame_num} ({phase}):"
                    })
                    
                    # Add the image
                    frames_for_api.append({
                        "image_path": str(overlay_path)
                    })
            
            # Add the main prompt
            prompt = self.create_comprehensive_grounding_prompt(
                video_metadata,
                all_objects,
                len(selected_frame_numbers)
            )
            
            frames_for_api.append({
                "text": prompt
            })
            
            # Call API with all frames
            print(f"\nCalling DASHSCOPE API with {len(selected_frame_numbers)} frames...")
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
                "raw_response": response,
                "timestamp": str(Path(base_path).stat().st_mtime)
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
        Process entire video to generate a single comprehensive grounded caption
        
        Args:
            dense_captions_json: Path to dense captions JSON
            video_id: Video identifier
            base_path: Base path for segmentation data
            output_dir: Output directory for results
            max_frames: Maximum frames to use
            model: Model name
            
        Returns:
            Result dictionary with comprehensive grounded caption
        """
        # Load video metadata
        print(f"Loading dense captions from {dense_captions_json}")
        video_metadata_dict = self.load_dense_captions_json(dense_captions_json)
        
        if video_id not in video_metadata_dict:
            raise ValueError(f"Video {video_id} not found in dense captions JSON")
        
        video_metadata = video_metadata_dict[video_id]
        
        print("\n" + "=" * 60)
        print(f"PROCESSING VIDEO: {video_id}")
        print("=" * 60)
        print(f"Scene: {video_metadata.get_scene_str()} {video_metadata.get_road_str()}")
        print(f"Conditions: {video_metadata.get_weather_str()}, {video_metadata.get_light_str()}")
        print(f"Total frames: {video_metadata.total_frames}")
        
        if video_metadata.accident:
            print(f"\nIncident Information:")
            print(f"  - Type: {video_metadata.causes}")
            print(f"  - Accident frame: {video_metadata.accident_frame}")
            print(f"  - Abnormal period: frames {video_metadata.abnormal_start}-{video_metadata.abnormal_end}")
            print(f"  - Safety measures: {video_metadata.measures}")
        
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
            output_file = output_path / f"{video_id}_comprehensive_grounded_caption.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved comprehensive result to: {output_file}")
            
            # Also save just the caption in a text file for easy reading
            caption_file = output_path / f"{video_id}_grounded_caption.txt"
            with open(caption_file, 'w') as f:
                f.write("COMPREHENSIVE GROUNDED CAPTION\n")
                f.write("=" * 60 + "\n\n")
                f.write(result['grounded_caption']['full_caption'])
                f.write("\n\n" + "=" * 60 + "\n")
                f.write(f"Total grounded objects: {len(result['grounded_caption']['objects'])}\n")
                f.write(f"Unique object types: {', '.join(result['grounded_caption']['unique_object_types'])}\n")
            print(f"Saved caption text to: {caption_file}")
            
            # Print summary
            print("\n" + "=" * 60)
            print("COMPREHENSIVE GROUNDED CAPTION GENERATED")
            print("=" * 60)
            print("\nCaption:")
            print("-" * 60)
            print(result['grounded_caption']['full_caption'])
            print("-" * 60)
            print(f"\nStatistics:")
            print(f"  - Frames analyzed: {result['processing_info']['frames_used']}/{result['processing_info']['total_available_frames']}")
            print(f"  - Total grounded objects in caption: {len(result['grounded_caption']['objects'])}")
            print(f"  - Unique object types grounded: {len(result['grounded_caption']['unique_object_types'])}")
            print(f"  - Object types in video: {len(result['processing_info']['all_object_types_in_video'])}")
            
            # Object frequency
            if result['grounded_caption']['object_counts']:
                print(f"\nObject frequency in caption:")
                for obj_type, count in sorted(result['grounded_caption']['object_counts'].items()):
                    print(f"  - {obj_type}: {count} occurrences")
            
            return result
        else:
            print("\nFailed to generate comprehensive grounded caption")
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
    
    # Path to dense captions JSON
    dense_captions_json = "dense_captions_output_1.json"
    
    # Video ID to process
    video_id = "002320"
    
    # Generate single comprehensive grounded caption for entire video
    print("\n" + "=" * 70)
    print(f"GENERATING COMPREHENSIVE GROUNDED CAPTION FOR VIDEO {video_id}")
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
            print(f"\n✓ Successfully generated comprehensive grounded caption")
            print(f"✓ Used {result['processing_info']['frames_used']} frames from the video")
            print(f"✓ Grounded {len(result['grounded_caption']['objects'])} object instances")
        else:
            print("\n✗ Failed to generate comprehensive grounded caption")
            
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()