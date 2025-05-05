import json
import os

# Use correct relative path
metadata_path = os.path.join("data", "raw", "00", "metadata.json")

if not os.path.exists(metadata_path):
    print("metadata.json NOT FOUND! Check the path.")
else:
    with open('data/raw/dfdc_train_part_00/metadata.json', 'r') as f:
        metadata=json.load(f)
    print("Metadata loaded successfully!")

    for video_name in list(metadata.keys())[:5]:
        label = metadata[video_name]['label']
        print(f"{video_name}: {label}")