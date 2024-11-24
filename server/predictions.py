import os
import logging
import requests
import time
from PIL import Image
from dotenv import load_dotenv
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

HF_API_URL = f"https://api-inference.huggingface.co/models/{os.getenv('HF_MODEL_ID')}"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}


def send_to_huggingface(image):
    """
    Send an image to the Hugging Face API for prediction.
    """
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data = {"inputs": image_data}

        logging.info("Sending image to Hugging Face API...")

        # Send the POST request with the base64-encoded image
        response = requests.post(HF_API_URL, headers=HEADERS, json=data, timeout=30)
        
        retry_count = 0
        max_retries = 5
        
        while response.status_code == 503 and retry_count < max_retries:
            response_json = response.json()
            estimated_time = response_json.get("estimated_time", 20)
            logging.warning(f"Model loading. Retrying after {estimated_time} seconds...")
           
            time.sleep(estimated_time)
            retry_count += 1
            response = requests.post(HF_API_URL, headers=HEADERS, json=data, timeout=30)

        if retry_count == max_retries:
            logging.error("Max retries reached. Model still unavailable.")
            return {"error": "Model loading took too long. Please try again later."}
        
        # Check response status
        if response.status_code != 200:
            logging.error(
                f"Error from Hugging Face API: {response.status_code} {response.text}"
            )
            return {"error": "Failed to get prediction from Hugging Face API"}
        
        # Parse and return the result
        result = response.json()
        logging.info(f"Prediction received: {result}")
        return result

    except Exception as e:
        logging.error(f"Error during Hugging Face API call: {e}")
        return {"error": str(e)}


def generate_prediction(image_stream):
    """
    Generate a prediction using the Hugging Face API.
    """
    try:
        # Load and prepare the image
        img = Image.open(image_stream).convert("RGB")

        # Send the image to Hugging Face for prediction
        result = send_to_huggingface(img)

        # Parse the result
        if "error" in result:
            return f"Error during prediction: {result['error']}"

        # Extract the label with the highest score
        predictions = sorted(result, key=lambda x: x["score"], reverse=True)
        top_prediction = predictions[0]
        label = top_prediction["label"]
        score = top_prediction["score"]

        return f"{label} with confidence {score:.2f}"

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return "Error during prediction"
