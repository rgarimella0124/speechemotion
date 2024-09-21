import os
import joblib
import soundfile
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"wav"}

# Load model with error handling
try:
    model = joblib.load("model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def extract_feature(file_name, mfcc, chroma, mel):
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma:
                stft = np.abs(librosa.stft(X))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
        return result
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    message = ""
    uploaded_file_path = None
    emotion = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part in the request. Please upload a file.", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected. Please select a file.", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            feature = extract_feature(filepath, mfcc=True, chroma=True, mel=True)
            if feature is not None and model is not None:
                try:
                    emotion = model.predict([feature])[0]
                    message = "File uploaded and processed successfully!"
                except Exception as e:
                    logging.error(f"Error predicting emotion: {e}")
                    message = "Error predicting emotion. Please try again."
            else:
                message = "Error processing file or model not loaded."
            uploaded_file_path = url_for("uploaded_file", filename=filename)
    return render_template("upload.html", emotion=emotion, message=message, uploaded_file_path=uploaded_file_path)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)