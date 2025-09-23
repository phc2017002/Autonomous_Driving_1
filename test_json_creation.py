import json

# IDs you want to keep
target_ids = {
    "002320", "003663", "004394", "005858", "006331", "006543", "007128",
    "007446", "008171", "010688", "013416", "003180", "004067", "004872",
    "005906", "006379", "006557", "007232", "007448", "009770", "011341",
    "013523"
}

# Input / Output paths
input_json = "cap_annotations.json"        # old JSON
output_json = "filtered_accident_data.json"  # new JSON

# Load old JSON
with open(input_json, "r") as f:
    data = json.load(f)

# Ensure data is a list
if isinstance(data, dict):
    data = [data]

# Filter by Video field
filtered_data = [entry for entry in data if str(entry.get("Video")) in target_ids]

# Save new JSON
with open(output_json, "w") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"Filtered {len(filtered_data)} entries out of {len(data)}")
print(f"Saved to {output_json}")
