import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
JPEG_CONTENT_TYPE = 'image/jpeg'


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))

    return model


def model_fn(model_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    return Image.open(io.BytesIO(request_body))


def predict_fn(request_body, model):
    testing_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))
        ])
    input_object = input_fn(request_body)
    input_object = testing_transform(input_object)
    input_object = input_object.unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_object)
    return prediction