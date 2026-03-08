import socketio
import eventlet
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import torch
import torch.nn as nn

# -----------------------------
# SocketIO & Flask Setup
# -----------------------------
sio = socketio.Server()
app = Flask(__name__)  # '__main__'
speed_limit = 10

# -----------------------------
# Nvidia Model Definition
# -----------------------------
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 100),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# -----------------------------
# Device & Model Loading
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NvidiaModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()  # important for inference

# -----------------------------
# Image Preprocessing
# -----------------------------
def img_preprocess(img):
    if img is None:
        return None
    if img.shape[0] < 135 or img.shape[1] < 1:
        return None
    try:
        img = img[60:135, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.resize(img, (200, 66))
        img = img / 255.0
        img = np.transpose(img, (2,0,1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img
    except Exception as e:
        print("Preprocessing error:", e)
        return None
# -----------------------------
# Steering Control
# -----------------------------
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# -----------------------------
# Telemetry Handler
# -----------------------------
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image_tensor = img_preprocess(image)
    
    if image_tensor is None:
        print("Invalid image received. Skipping frame.")
        send_control(0, 0)
        return

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        steering_angle = model(image_tensor).item()

    throttle = 1.0 - speed / speed_limit
    print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}")
    send_control(steering_angle, throttle)

# -----------------------------
# Connection Handler
# -----------------------------
@sio.on('connect')
def connect(sid, environ):
    print('Connected:', sid)
    send_control(0, 0)

# -----------------------------
# Start Server
# -----------------------------
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)