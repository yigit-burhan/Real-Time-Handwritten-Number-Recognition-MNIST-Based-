import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --------------------
# Model definition (MUST match training)
# --------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --------------------
# Load trained model
# --------------------
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# --------------------
# Image preprocessing (MNIST-style)
# --------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --------------------
# Load image
# --------------------
img = Image.open("digit.png")
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension

# --------------------
# Prediction
# --------------------
with torch.no_grad():
    output = model(img)
    prediction = output.argmax(1).item()

print("Predicted digit:", prediction)
