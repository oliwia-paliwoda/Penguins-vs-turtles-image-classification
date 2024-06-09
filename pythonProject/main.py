import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np


# Funkcja do przetwarzania obrazu przed podaniem go do modelu
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    # Konwersja obrazu PIL na tensor i normalizacja
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    # Dodanie wymiaru wsadowego
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


# Funkcja do klasyfikacji obrazu za pomocą modelu
def classify_image(image):
    # Przetworzenie obrazu
    input_tensor = preprocess_image(image)
    # Sprawdzenie dostępności urządzenia GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Zdefiniowanie i załadowanie modelu ResNet18
    model_resnet18 = models.resnet18(weights=None)
    model_resnet18.fc = torch.nn.Linear(model_resnet18.fc.in_features,
                                        2)  # Zakładamy, że model ma 2 klasy (Pingwin i Żółw)
    checkpoint_resnet18 = torch.load('model_resnet18.pth', map_location=device)
    model_resnet18.load_state_dict(checkpoint_resnet18['model_state_dict'])
    model_resnet18.to(device)
    model_resnet18.eval()

    # Zdefiniowanie i załadowanie modelu VGG16
    model_vgg16 = models.vgg16(weights=None)
    num_features = model_vgg16.classifier[6].in_features
    model_vgg16.classifier[6] = torch.nn.Linear(num_features, 2)  # Zakładamy, że model ma 2 klasy (Pingwin i Żółw)
    checkpoint_vgg16 = torch.load('model_vgg16.pth', map_location=device)
    model_vgg16.load_state_dict(checkpoint_vgg16['model_state_dict'])
    model_vgg16.to(device)
    model_vgg16.eval()

    # Klasyfikacja obrazu za pomocą modelu ResNet18
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output_resnet18 = model_resnet18(input_tensor)
    _, predicted_resnet18 = torch.max(output_resnet18, 1)

    # Klasyfikacja obrazu za pomocą modelu VGG16
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output_vgg16 = model_vgg16(input_tensor)
    _, predicted_vgg16 = torch.max(output_vgg16, 1)

    # Zwracanie predykcji z obu modeli
    prediction_resnet18 = "Penguin" if predicted_resnet18.item() == 0 else "Turtle"
    prediction_vgg16 = "Penguin" if predicted_vgg16.item() == 0 else "Turtle"
    return prediction_resnet18, prediction_vgg16


# Interfejs Gradio
input_component = gr.components.Image(label="Wczytaj obraz")
output_component_resnet18 = gr.components.Label(label="Predykcja z ResNet18")
output_component_vgg16 = gr.components.Label(label="Predykcja z VGG16")

gr.Interface(fn=classify_image, inputs=input_component, outputs=[output_component_resnet18, output_component_vgg16]).launch(server_name="0.0.0.0", server_port=8081)
