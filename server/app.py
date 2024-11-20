from flask import Flask, request, render_template, flash, redirect, url_for
from dotenv import load_dotenv
import os
import io
import base64
import logging
from predictions import generate_prediction

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        # Render the page with no prediction or image initially
        return render_template("home.html", prediction=None, image_data=None)

    if request.method == "POST":
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(url_for('home'))

        file = request.files['image']

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(url_for('home'))

        if file:
            try:
                image_stream = io.BytesIO(file.read())
                logging.info("Starting prediction process...")

                # Generate the prediction
                predicted_label = generate_prediction(image_stream)
                logging.info(f"Prediction completed: {predicted_label}")

                # Reset and encode image for rendering
                image_stream.seek(0)
                img_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
                logging.info("Image successfully encoded to base64.")

                # Render the same template with prediction and image data
                return render_template(
                    "home.html",
                    prediction=predicted_label,
                    image_data=img_base64
                )

            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                flash(f"Error during prediction: {str(e)}", "danger")
                return redirect(url_for('home'))

        flash('Prediction failed', 'danger')
        return redirect(url_for('home'))


