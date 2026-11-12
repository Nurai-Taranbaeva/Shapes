import torch
import torch.nn as nn
import io
import uvicorn

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms

app = FastAPI(
    title="Shapes Project",
    description="Image Classification"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parent / "model (19).pth"


class ShapeCNN(nn.Module):
    def __init__(self, num_classes):
        super(ShapeCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = ShapeCNN(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


@app.get("/")
def home():
    return {
        "message": "Shapes Recognition API",
        "status": "running"
    }


@app.get("/classes")
def get_classes():
    return {"classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Only JPG and PNG images are supported")

    contents = await file.read()

    if not contents:
        raise HTTPException(400, "Empty file")

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "File is too large")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid image")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)