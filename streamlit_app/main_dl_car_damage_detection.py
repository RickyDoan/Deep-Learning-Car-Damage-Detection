import streamlit as st
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
import torch

# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    best_params_ResNet = {'dropout_rate': 0.37112580272910733, 'lr': 0.0010867605115576732}
    def __init__(self, num_classes=6, dropout_rate=best_params_ResNet['dropout_rate']):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Function to load the pretrained model
@st.cache_resource
def load_model():
    # Number of classes (Update this based on your problem)
    num_classes = 6

    # Load the model architecture
    model = CarClassifierResNet(num_classes=num_classes)

    # Load the pretrained weights
    model.load_state_dict(torch.load("model_resnet.pth", map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    return model


# Get the class labels (Update these names with your dataset's class labels)
@st.cache_data
def get_class_labels():
    return ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']


# Preprocess the input image
def preprocess_image(image):
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match the model's input size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per pretrained model
    ])
    return transformer(image).unsqueeze(0)  # Add batch dimension


# Predict the damage category
def predict_damage(image, model, class_labels):
    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)  # Forward pass through the model
        _, prediction = torch.max(outputs, 1)  # Get the predicted class index
        predicted_label = class_labels[prediction.item()]
    return predicted_label


# Streamlit UI starts here
st.title("Car Damage Detection with Deep Learning")
st.write("Upload a car image to detect the type of damage.")

# Load the pretrained model and class labels
model = load_model()
class_labels = get_class_labels()

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Automatically predict when an image is uploaded
    st.write("Predicting...")
    prediction = predict_damage(image, model, class_labels)
    st.success(f"Prediction: {prediction}")