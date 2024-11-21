import os
import logging
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

# Define transformations for the input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    try:
        # Get the model file name from environment variables
        model_file = os.getenv("MODEL")
        if not model_file:
            raise ValueError("Environment variable 'MODEL' is not set or empty.")

        # Construct the absolute path to the model
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_directory, "models", model_file)
        
        logging.info(f"Loading model from: {model_path}")
        
        # Check if the model path exists
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at path: {model_path}")
            raise FileNotFoundError(f"Model file not found at path: {model_path}")

        # Load the model
        model = resnet50(num_classes=4)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()

        # Set up the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logging.info("Model loaded successfully.")
        return model, device
    
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
