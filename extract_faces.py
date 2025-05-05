import os
import cv2
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# Input and output directories
input_dir = 'data/raw/dfdc_train_part_02/dfdc_train_part_2'
output_dir = 'processed/frames'
os.makedirs(output_dir, exist_ok=True)

# Get all video files
video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
print(f"[INFO] Found {len(video_files)} video files.")

# Frame sampling interval
frame_interval = 5  # You can decrease this to reduce extracted frames

# Process each video
for video_name in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join(input_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    face_resized = cv2.resize(face_crop, (224, 224))
                    face_filename = f"{os.path.splitext(video_name)[0]}_frame{frame_num}_face{i}.jpg"
                    save_path = os.path.join(output_dir, face_filename)
                    cv2.imwrite(save_path, face_resized)
                    saved_count += 1
                    print(f"[SAVED] {save_path}")

        frame_num += 1

    cap.release()
    print(f"[DONE] {video_name}: Total frames: {frame_num}, Saved faces: {saved_count}")

print("[INFO] All videos processed.")