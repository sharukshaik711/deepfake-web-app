import os
import json

# Read metadata from one folder (try with '00' first)
metadata_path = os.path.join("00", "metadata.json")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Print the first 5 entries
for video_name in list(metadata.keys())[:5]:
    label = metadata[video_name]['label']
    print(f"{video_name}: {label}")