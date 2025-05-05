import os
import cv2
import torch
import numpy as np
import warnings
from torchvision import transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("../models/shahid_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Get latest video from ../outputs
video_folder = "../outputs"
videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
if not videos:
    raise FileNotFoundError("No .mp4 videos found in ../outputs folder.")
latest_video = max(videos, key=lambda x: os.path.getctime(os.path.join(video_folder, x)))
video_path = os.path.join(video_folder, latest_video)
print(f"\nProcessing video: {video_path}")

# Read video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

confidences = []
progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        fake_confidence = probs[0][1].item()
        confidences.append(fake_confidence)

    progress_bar.update(1)

cap.release()
progress_bar.close()

# Plot frame-wise fake confidence
plt.plot(confidences)
plt.axhline(y=0.95, color='r', linestyle='--', label='Fake Threshold (0.95)')
plt.title(f"Frame-wise Fake Confidence: {latest_video}")
plt.xlabel("Frame Index")
plt.ylabel("Fake Confidence")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.splitext(video_path)[0] + "_confidence_plot.png")
plt.show()

# Prediction logic
avg_confidence = np.mean(confidences)
if any(conf > 0.95 for conf in confidences):
    prediction = "FAKE"
elif avg_confidence < 0.5:
    prediction = "REAL"
else:
    prediction = "FAKE"

# Output result
result_str = (
    f"Video: {os.path.basename(video_path)}\n"
    f"Prediction: {prediction}\n"
    f"Average Fake Confidence: {avg_confidence:.4f}\n"
    f"Frames Processed: {len(confidences)}\n"
)

print("\n" + "="*40)
print(result_str)
print("="*40)

# Save result to a .txt file
output_txt = os.path.splitext(video_path)[0] + "_prediction.txt"
with open(output_txt, "w") as f:
    f.write(result_str)

print(f"Prediction saved to: {output_txt}")