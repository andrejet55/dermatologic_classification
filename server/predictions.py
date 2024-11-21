import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define transformations for the input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_path):
    """
    Load the PyTorch model from the specified path.
    This function is called once at server startup.
    """
    try:
        logging.info(f"Loading the model from {model_path}...")
        model = resnet50(num_classes=4)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logging.info("Model loaded successfully.")
        return model, device
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Please check the path.")
        raise
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        raise

def generate_prediction(image_stream, model, device):
    """
    Generate a prediction for the given image using the preloaded model.
    """
    try:
        logging.info("Starting the prediction process...")

        # Load and preprocess the image from the stream
        img = Image.open(image_stream).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(device)
        logging.info("Image preprocessing completed.")

        # Predict the class of the image
        with torch.no_grad():
            logging.info("Running the prediction on the image...")
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
        logging.info(f"Prediction completed. Predicted index: {predicted_idx.item()}")

        # Categories for classification
        categories = ["Level 0", "Level 1", "Level 2", "Level 3"]
        predicted_label = categories[predicted_idx.item()]
        logging.info(f"Final predicted label: {predicted_label}")

        return predicted_label
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return "Error during prediction"
