import os
import json
import base64
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Any

class DenseCaptionGenerator:
    def __init__(self, api_key: str = None, image_dir: str = "images", json_path: str = "data.json"):
        """
        Initialize the Dense Caption Generator
        
        Args:
            api_key: API key for DashScope/Model Studio
            image_dir: Directory containing video frame folders
            json_path: Path to the JSON file with video metadata
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
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
    
    def parse_metadata(self, entry: Dict[str, Any]) -> str:
        """Parse JSON entry into readable context"""
        weather = self.weather_map.get(entry.get("weather(sunny,rainy,snowy,foggy)1-4", 1))
        light = self.light_map.get(entry.get("light(day,night)1-2", 1))
        scene = self.scene_map.get(entry.get("scenes(highway,tunnel,mountain,urban,rural)1-5", 4))
        road_type = self.road_map.get(entry.get("linear(arterials,curve,intersection,T-junction,ramp) 1-5", 3))
        
        context = f"""
Scene Context:
- Weather: {weather}
- Lighting: {light}
- Environment: {scene}
- Road Type: {road_type}

Accident Information:
- Accident Occurred: {'Yes' if entry.get('whether an accident occurred (1/0)', 0) == 1 else 'No'}
- Abnormal Behavior Start Frame: {entry.get('abnormal start frame', 'N/A')}
- Abnormal Behavior End Frame: {entry.get('abnormal end frame', 'N/A')}
- Accident Frame: {entry.get('accident frame', 'N/A')}
- Total Frames: {entry.get('total frames', 'N/A')}

Event Description: {entry.get('texts', 'N/A')}
Accident Cause: {entry.get('causes', 'N/A')}
Preventive Measures: {entry.get('measures', 'N/A')}
"""
        return context
    
    def generate_dense_caption_prompt(self, context: str) -> str:
        """Create a comprehensive prompt for dense caption generation"""
        prompt = f"""Based on the following frames and context information, generate a dense, detailed caption that describes:

1. The visual scene in detail (vehicles, road conditions, environment)
2. The sequence of events leading to the incident
3. Key safety observations
4. Temporal progression of the situation

Context Information:
{context}

Please provide a dense caption that:
- Describes all visible objects, their positions, and movements
- Explains the traffic situation and road conditions
- Details the sequence of events chronologically
- Highlights safety-critical elements
- Includes spatial relationships between vehicles and road elements
- Mentions any potential hazards or risk factors

Generate a comprehensive, detailed caption for the entire video:"""
        return prompt
    
    def get_video_frames(self, video_folder: str) -> List[Path]:
        """
        Get a list of frame images from a video folder
        
        Args:
            video_folder: Name of the folder containing frames
            max_frames: Maximum frames to sample for captioning
        
        Returns:
            List of image paths
        """
        folder_path = self.image_dir / video_folder + "/images"
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Warning: Video folder {video_folder} not found")
            return []
        
        # List all jpg/png frames
        frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        
        if not frames:
            print(f"Warning: No frames found in {video_folder}")
            return []
        
        
        return frames
    
    def generate_caption_for_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dense caption for an entire video by analyzing multiple frames
        """
        video_folder = entry.get("Video", "")
        context = self.parse_metadata(entry)
        prompt = self.generate_dense_caption_prompt(context)
        
        # Load multiple frames
        image_paths = self.get_video_frames(video_folder, max_frames=12)
        
        if not image_paths:
            print(f"Warning: No frames found for {video_folder}")
            return {**entry, "dense_caption": "No frames available for caption generation"}
        
        message_content = []
        
        # Add sampled frames
        for img_path in image_paths:
            try:
                image_url = self.create_local_image_url(img_path)
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
                print(f"  Added frame: {img_path.name}")
            except Exception as e:
                print(f"  Error loading frame {img_path}: {e}")
        
        # Add text prompt
        message_content.append({"type": "text", "text": prompt})
        
        try:
            # API call
            completion = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{"role": "user", "content": message_content}],
                max_tokens=700,
                temperature=0.7
            )
            
            caption = completion.choices[0].message.content
            print(f"  Generated caption for {video_folder}")
            
            return {
                **entry,
                "dense_caption": caption,
                "frames_used": [str(p.name) for p in image_paths]
            }
        
        except Exception as e:
            print(f"  Error generating caption: {e}")
            return {
                **entry,
                "dense_caption": f"Error generating caption: {str(e)}",
                "frames_used": [str(p.name) for p in image_paths]
            }
    
    def process_json_file(self, output_path: str = "output_with_captions.json"):
        """
        Process entire JSON file and generate captions for all entries
        
        Args:
            output_path: Path to save the output JSON with captions
        """
        # Load JSON data
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single entry and list of entries
        if isinstance(data, dict):
            entries = [data]
        else:
            entries = data
        
        results = []
        total = len(entries)
        
        print(f"Processing {total} entries...")
        print("=" * 50)
        
        for idx, entry in enumerate(entries, 1):
            print(f"\nProcessing entry {idx}/{total}: {entry.get('Video', 'Unknown')}")
            result = self.generate_caption_for_entry(entry)
            results.append(result)
            
            # Save intermediate results every 5 entries
            if idx % 5 == 0:
                with open(f"{output_path}.tmp", 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  Saved intermediate results ({idx}/{total})")
        
        # Save final results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 50)
        print(f"Processing complete! Results saved to {output_path}")
        
        return results


# Example usage
def main():
    # Initialize generator
    generator = DenseCaptionGenerator(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # Or provide directly
        image_dir="images",                      # Directory containing video frame folders
        json_path="filtered_accident_data.json"           # Path to your JSON file
    )
    
    # Process entire JSON file
    results = generator.process_json_file(output_path="dense_captions_output.json")
    
    # Or test with single entry
    test_entry = {
        "Video": "000001",
        "weather(sunny,rainy,snowy,foggy)1-4": 1,
        "light(day,night)1-2": 1,
        "scenes(highway,tunnel,mountain,urban,rural)1-5": 4,
        "linear(arterials,curve,intersection,T-junction,ramp) 1-5": 3,
        "type": 10,
        "whether an accident occurred (1/0)": 1,
        "abnormal start frame": 18,
        "abnormal end frame": 50,
        "accident frame": 34,
        "total frames": 50,
        "[0,tai]": 18,
        "[tai,tco]": 16,
        "[tai,tae]": 32,
        "[tco,tae]": 16,
        "[tae,end]": 0,
        "texts": "[CLS]a vehicle goes straight at signalized junctions[SEP]",
        "causes": "The ego-car's vision is blocked or blurred, and there is no time to brake",
        "measures": "Ego-cars should slow down or honk their horns when they stop at intersections or trunk roads where their vision is blocked to prevent other vehicles or pedestrians from rushing out suddenly."
    }
    
    # caption = generator.generate_caption_for_entry(test_entry)
    print("\nGenerated Dense Caption:")
    print(caption["dense_caption"])


if __name__ == "__main__":
    main()
