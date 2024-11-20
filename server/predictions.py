import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_prediction(image_stream):
    logging.info("Starting the prediction process...")

    # Define transformations for the input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    logging.info("Image preprocessing pipeline created.")

    # Load the trained model
    def load_model(model_path):
        try:
            logging.info(f"Loading the model from {model_path}...")
            model = resnet50(num_classes=4)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
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

    # Load and preprocess the image from a stream
    def get_img_array(image_stream, device):
        try:
            logging.info("Preprocessing the input image...")
            img = Image.open(image_stream).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(device)
            logging.info("Image preprocessing completed.")
            return img_tensor, img
        except Exception as e:
            logging.error(f"Error preprocessing the image: {e}")
            raise

    # Predict the class of an image
    def predict_image(model, img_tensor, categories):
        try:
            logging.info("Running the prediction on the image...")
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted_idx = torch.max(output, 1)
            logging.info(f"Prediction completed. Predicted index: {predicted_idx.item()}")
            return categories[predicted_idx.item()]
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    # Categories for classification (adjust as per your dataset)
    categories = ["Level 0", "Level 1", "Level 2", "Level 3"]

    # Path to your saved model (update this path as needed)
    model_path = "./models/best_resnet50_model.pth"

    try:
        # Load the model and device
        model, device = load_model(model_path)

        # Load and preprocess the image
        img_tensor, _ = get_img_array(image_stream, device)

        # Make a prediction
        predicted_label = predict_image(model, img_tensor, categories)

        logging.info(f"Final predicted label: {predicted_label}")
        return predicted_label

    except Exception as e:
        logging.error(f"An error occurred in the prediction process: {e}")
        return "Error during prediction"
