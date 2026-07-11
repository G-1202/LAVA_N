from PIL import Image
from torchvision import transforms
from .AE_network import AEModel
import torch
import numpy as np
import cv2


model = AEModel(input_features=8400*85, output_features=33)
model.load_state_dict(torch.load('AE/model.ckpt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image


def predict(image_path):
    image = preprocess_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return output.cpu().numpy()

