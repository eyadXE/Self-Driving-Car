# 🚗 Self-Driving Car with PyTorch and Flask

A self-driving car simulation using behavioral cloning with PyTorch. The system receives images from the **Udacity Self-Driving Car Simulator**, predicts steering angles in real time, and controls the vehicle via a Flask + SocketIO backend server. The model is based on **NVIDIA's end-to-end CNN architecture** for autonomous driving.

---
## DEMO
![Demo](DEMO.gif)
## 📐 Model Architecture (NVIDIA CNN)

Input: **66×200 RGB image** → Output: **Steering angle (continuous value)**

| Layer | Parameters | Output Shape | Activation |
|---|---|---|---|
| Conv2D | 24 filters, 5×5, stride=2 | 31×98×24 | ELU |
| Conv2D | 36 filters, 5×5, stride=2 | 14×47×36 | ELU |
| Conv2D | 48 filters, 5×5, stride=2 | 5×22×48 | ELU |
| Conv2D | 64 filters, 3×3, stride=1 | 1×18×64 | ELU |
| Flatten | — | 1152 | — |
| Fully Connected | 100 neurons | 100 | ELU |
| Fully Connected | 50 neurons | 50 | ELU |
| Fully Connected | 10 neurons | 10 | ELU |
| Fully Connected | 1 neuron | 1 | Linear |

**Activation:** ELU (Exponential Linear Unit) for faster convergence and better gradient flow.

---

## 🖼️ Image Preprocessing

Each frame is preprocessed before being passed to the network:

```python
img = img[60:135, :, :]                        # Crop sky and hood
img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)     # RGB → YUV
img = cv2.GaussianBlur(img, (3, 3), 0)         # Reduce noise
img = cv2.resize(img, (200, 66))               # Resize to NVIDIA input dims
img = img / 255.0                              # Normalize to [0, 1]
img = np.transpose(img, (2, 0, 1))            # HWC → CHW for PyTorch
```

---

## 📊 Training

| Setting | Value |
|---|---|
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam (lr = 1e-3) |
| Batch Size | 100 |
| Epochs | 10 |
| Train / Val / Test Split | 80% / 10% / 10% |

**Data augmentation applied:** brightness adjustment, flipping, panning, and zoom.  
**GPU recommended** for training.

---

## 🐳 Docker Setup

Docker ensures consistent deployment without dependency issues.

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 4567
CMD ["python", "drive.py"]
```

**Build & Run:**
```bash
# Build
docker build -t selfdriving-car .

# Run
docker run --rm -p 4567:4567 selfdriving-car
```

> Port `4567` is mapped to the host for Udacity simulator communication.  
> For faster inference, add GPU support via [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).

---

## 🖥️ Running the Project

### Prerequisites
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) installed
- Docker (or Python + virtual environment)
- Trained model file: `model.pth`

### Option 1 — Docker
```bash
docker build -t selfdriving-car .
docker run --rm -p 4567:4567 selfdriving-car
```

### Option 2 — Local Python
```bash
conda activate carenv   # or your virtualenv
python drive.py
```

Then open the Udacity Simulator, select **Autonomous Mode**, and connect to `localhost:4567`.

---

## ⚙️ Flask + SocketIO Server

The server handles the full inference loop:

1. Receives telemetry (image + speed) from the simulator
2. Preprocesses the image
3. Runs inference with the PyTorch model
4. Computes throttle based on current speed
5. Sends steering angle and throttle back to the simulator

```python
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image).to(device)

    with torch.no_grad():
        steering_angle = model(image).item()
    throttle = 1.0 - speed / speed_limit
    send_control(steering_angle, throttle)
```

---

## 💡 Tips

- Ensure your model architecture matches **exactly** between training and inference — otherwise `load_state_dict` will fail.
- Always call `model.eval()` before inference.
- Improve accuracy with:
  - More data augmentation (shadows, rotations)
  - Deeper CNN architecture
  - Dropout layers to prevent overfitting
  - Learning rate schedulers

---

## 📚 References

- [NVIDIA — End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Flask-SocketIO Docs](https://flask-socketio.readthedocs.io/)
