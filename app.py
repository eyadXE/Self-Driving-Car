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
sio = socketio.Server()
app = Flask(__name__)

speed_limit = 20
class PilotNet(nn.Module):

    def __init__(self):
        super(PilotNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),

            nn.Conv2d(48, 64, 3),
            nn.ELU(),

            nn.Conv2d(64, 64, 3),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 18 * 1, 100),
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PilotNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
def img_preprocess(img):

    if img is None:
        return None

    try:
        # crop (remove sky + car hood)
        img = img[60:135, :, :]

        # RGB → YUV (important for driving)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        # blur
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # resize
        img = cv2.resize(img, (200, 66))

        # normalize EXACTLY like training
        img = img / 127.5 - 1.0

        # CHW format
        img = np.transpose(img, (2, 0, 1))

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img

    except Exception as e:
        print("Preprocess error:", e)
        return None
def send_control(steering_angle, throttle):

    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })
@sio.on('telemetry')
def telemetry(sid, data):

    speed = float(data['speed'])

    # decode image
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)

    # preprocess
    image_tensor = img_preprocess(image)

    if image_tensor is None:
        print("Invalid frame")
        send_control(0.0, 0.0)
        return

    image_tensor = image_tensor.to(device)

    # inference
    with torch.no_grad():
        steering_angle = model(image_tensor).item()

    # throttle control
    throttle = 1.0 - (speed / speed_limit)

    print(f"Steering: {steering_angle:.4f} | Speed: {speed:.2f} | Throttle: {throttle:.2f}")

    send_control(steering_angle, throttle)
@sio.on('connect')
def connect(sid, environ):
    print("Connected:", sid)
    send_control(0.0, 0.0)
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)