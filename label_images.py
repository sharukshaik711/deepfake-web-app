import os
import json
import shutil

# Paths
raw_data_path = 'data/raw'
frames_path = 'processed/frames'
output_real = 'processed/train/real'
output_fake = 'processed/train/fake'

# Make sure destination folders exist
os.makedirs(output_real, exist_ok=True)
os.makedirs(output_fake, exist_ok=True)

# Read all frame filenames once
frame_files = os.listdir(frames_path)

# Loop through each dfdc_train_part folder
for part_folder in os.listdir(raw_data_path):
    part_path = os.path.join(raw_data_path, part_folder)

    # If there's an inner folder, go inside it
    if len(os.listdir(part_path)) == 1:
        inner_folder = os.listdir(part_path)[0]
        part_path = os.path.join(part_path, inner_folder)

    metadata_file = os.path.join(part_path, 'metadata.json')

    if not os.path.exists(metadata_file):
        print(f"Metadata not found in {part_path}")
        continue

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # For every video in metadata
    for video_filename, info in metadata.items():
        label = info['label']  # REAL or FAKE
        video_name = os.path.splitext(video_filename)[0]  # remove '.mp4'

        found_frame = False

        # Now match frames properly
        for frame_file in frame_files:
            if frame_file.startswith(video_name + "_frame"):
                src_path = os.path.join(frames_path, frame_file)

                # Decide output folder
                if label == 'REAL':
                    dest_path = os.path.join(output_real, frame_file)
                else:
                    dest_path = os.path.join(output_fake, frame_file)

                shutil.copy(src_path, dest_path)
                found_frame = True

        if not found_frame:
            print(f"No frames found for video {video_name}")

print("\n✅ Done! Frames copied into real/fake folders correctly!")