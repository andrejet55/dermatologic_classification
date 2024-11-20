import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt


def generate_prediction(image_path):
    # Define transformations for the input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    ])

    # Load the trained model
    def load_model(model_path):
        # model = torch.load(model_path)
        # Instantiate the ResNet50 model
        # Adjust num_classes to match your categories
        model = resnet50(num_classes=4)

        # Load the saved state dictionary into the model
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, device

    # Function to load and preprocess the image
    def get_img_array(img_path, device):
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(device)
        return img_tensor, img

    # Predict the class of an image
    def predict_image(model, img_tensor, categories):
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
        return categories[predicted_idx.item()]

    # Function to display the image with the predicted label
    def display_image_with_label(img, label):
        plt.imshow(img)
        plt.title(f"Predicted: {label}")
        plt.axis("off")
        plt.show()

    # Path to your saved model and test image
    model_path = "/models/best_resnet50_model.pth"
    # image_path = "/path/levle0_0.jpg"

    # Categories for classification
    # Adjust these as per your dataset
    categories = ["Level 0", "Level 1", "Level 2", "Level 3"]

    # Load the model and device
    model, device = load_model(model_path)

    # Load and preprocess the image
    img_tensor, original_img = get_img_array(image_path, device)

    # Make a prediction
    predicted_label = predict_image(model, img_tensor, categories)
    print(predicted_label)

    # Display the image with its predicted label
    display_image_with_label(original_img, predicted_label)

    return predicted_label
