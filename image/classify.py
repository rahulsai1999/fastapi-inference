import io
import json
import pathlib

from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

# file path to imagenet_class_index.json
file_path = "imagenet_class_index.json"
imagenet_class_index = json.load(open(pathlib.Path(__file__).parent / file_path))

model = models.densenet121(weights="IMAGENET1K_V1")
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
