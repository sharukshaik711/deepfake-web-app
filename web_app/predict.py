import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import warnings
import subprocess
import uuid

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model_path = os.path.join("models", "shahid_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def compress_video(input_path):
    """
    Compress and resize the video using ffmpeg to 360p for faster processing.
    Returns path to compressed file.
    """
    output_path = f"{os.path.splitext(input_path)[0]}compressed{uuid.uuid4().hex}.mp4"
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=640:360",  # Resize to 360p
        "-c:v", "libx264", "-preset", "fast", "-crf", "28",
        "-c:a", "copy",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        raise RuntimeError("FFmpeg compression failed")
    return output_path

def predict_video(video_path, frame_skip=10):
    """
    Predict whether video is REAL or FAKE using EfficientNet-B0.
    """
    compressed_path = compress_video(video_path)

    cap = cv2.VideoCapture(compressed_path)
    confidences = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            fake_confidence = probs[0][1].item()
            confidences.append(fake_confidence)

        frame_index += 1

    cap.release()

    # Delete compressed file to save space
    os.remove(compressed_path)

    if not confidences:
        return "UNKNOWN", 0.0, 0

    avg_conf = np.mean(confidences)
    prediction = "FAKE" if any(conf > 0.95 for conf in confidences) else "REAL"

    return prediction, avg_conf, len(confidences)