import torch
from PIL import Image
from torchvision import transforms
from train_mnist import CNN

model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open("digit.png")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    prediction = output.argmax(1).item()

print("Predicted digit:", prediction)
