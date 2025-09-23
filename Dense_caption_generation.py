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
            image_dir: Directory containing video frame images
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
        prompt = f"""Based on the image and the following context information, generate a dense, detailed caption that describes:

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

Generate a comprehensive, detailed caption:"""
        return prompt
    
    def get_frame_images(self, video_name: str, entry: Dict[str, Any]) -> List[Path]:
        """
        Get relevant frame images for the video
        
        Args:
            video_name: Base name of the video
            entry: JSON entry with frame information
        
        Returns:
            List of image paths to process
        """
        frames_to_load = []
        base_name = video_name.replace('.jpg', '').replace('.png', '')
        
        # Key frames to analyze
        key_frames = [
            entry.get('abnormal start frame'),
            entry.get('accident frame'),
            entry.get('abnormal end frame')
        ]
        
        # Add the main frame
        main_frame = self.image_dir / video_name
        if main_frame.exists():
            frames_to_load.append(main_frame)
        
        # Try to load key frames if they exist
        for frame_num in key_frames:
            if frame_num:
                # Try different naming conventions
                possible_names = [
                    f"{base_name}_{frame_num:03d}.jpg",
                    f"{base_name}_{frame_num}.jpg",
                    f"{base_name}_frame_{frame_num}.jpg",
                    f"frame_{frame_num}.jpg"
                ]
                
                for name in possible_names:
                    frame_path = self.image_dir / name
                    if frame_path.exists() and frame_path not in frames_to_load:
                        frames_to_load.append(frame_path)
                        break
        
        return frames_to_load[:3]  # Limit to 3 images max for API constraints
    
    def generate_caption_for_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dense caption for a single JSON entry
        
        Args:
            entry: Single entry from JSON file
        
        Returns:
            Dictionary with original entry and generated caption
        """
        video_name = entry.get("Video", "")
        context = self.parse_metadata(entry)
        prompt = self.generate_dense_caption_prompt(context)
        
        # Get relevant images
        image_paths = self.get_frame_images(video_name, entry)
        
        if not image_paths:
            print(f"Warning: No images found for {video_name}")
            return {**entry, "dense_caption": "No image available for caption generation"}
        
        # Prepare message content
        message_content = []
        
        # Add images
        for img_path in image_paths:
            try:
                image_url = self.create_local_image_url(img_path)
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
                print(f"  Added image: {img_path.name}")
            except Exception as e:
                print(f"  Error loading image {img_path}: {e}")
        
        # Add text prompt
        message_content.append({"type": "text", "text": prompt})
        
        try:
            # Call API
            completion = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{
                    "role": "user",
                    "content": message_content
                }],
                max_tokens=500,
                temperature=0.7
            )
            
            caption = completion.choices[0].message.content
            print(f"  Generated caption successfully")
            
            return {
                **entry,
                "dense_caption": caption,
                "images_used": [str(p.name) for p in image_paths]
            }
            
        except Exception as e:
            print(f"  Error generating caption: {e}")
            return {
                **entry,
                "dense_caption": f"Error generating caption: {str(e)}",
                "images_used": [str(p.name) for p in image_paths]
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
            
            # Optional: Save intermediate results
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
    
    def generate_single_caption(self, json_entry: Dict[str, Any], image_path: str = None) -> str:
        """
        Generate caption for a single entry (useful for testing)
        
        Args:
            json_entry: Single JSON entry
            image_path: Optional specific image path to use
        
        Returns:
            Generated dense caption
        """
        if image_path:
            # Use specific image
            context = self.parse_metadata(json_entry)
            prompt = self.generate_dense_caption_prompt(context)
            
            image_url = self.create_local_image_url(Path(image_path))
            
            completion = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=500,
                temperature=0.7
            )
            
            return completion.choices[0].message.content
        else:
            # Use default logic
            result = self.generate_caption_for_entry(json_entry)
            return result.get("dense_caption", "")


# Example usage
def main():
    # Initialize generator
    generator = DenseCaptionGenerator(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # Or provide directly
        image_dir="images",  # Directory containing video frames
        json_path="accident_data.json"  # Path to your JSON file
    )
    
    # Process entire JSON file
    results = generator.process_json_file(output_path="dense_captions_output.json")
    
    # Or test with single entry
    test_entry = {
        "Video": "000001.jpg",
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
    
    # Generate caption for single entry
    caption = generator.generate_single_caption(test_entry)
    print("\nGenerated Dense Caption:")
    print(caption)


if __name__ == "__main__":
    main()