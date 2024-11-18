from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os

load_dotenv() 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = os.getenv("SECRET_KEY")

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('home'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('File successfully uploaded', 'success')
        return redirect(url_for('home'))

    flash('File upload failed', 'danger')
    return redirect(url_for('home'))
