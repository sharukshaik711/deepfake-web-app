import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import warnings

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    confidences = []

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

    cap.release()

    avg_conf = np.mean(confidences)
    prediction = "FAKE" if any(conf > 0.95 for conf in confidences) else "REAL"

    return prediction, avg_conf, len(confidences)