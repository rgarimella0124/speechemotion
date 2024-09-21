import os
import pickle
import soundfile
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"wav"}

# Load model with error handling
model = None
try:
    model_path = os.path.join(os.path.dirname(__file__), "emotion_model.pkl")
    logging.info(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file does not exist at: {model_path}")
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    logging.error(traceback.format_exc())

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def extract_feature(file_name, mfcc, chroma, mel):
    try:
        logging.info(f"Extracting features from: {file_name}")
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
        logging.info("Features extracted successfully")
        return result
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        logging.error(traceback.format_exc())
        return None

def predict_emotion(file_path):
    try:
        logging.info(f"Predicting emotion for file: {file_path}")
        if model is None:
            raise ValueError("Model not loaded. Please check if 'emotion_model.pkl' exists and is valid.")
        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        if features is None:
            raise ValueError("Failed to extract features")
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        logging.info(f"Prediction successful: {prediction[0]}")
        return prediction[0]
    except Exception as e:
        logging.error(f"Error predicting emotion: {e}")
        logging.error(traceback.format_exc())
        raise

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        logging.info("Received POST request for file upload")
        if "file" not in request.files:
            logging.warning("No file part in the request")
            return jsonify({"error": "No file part in the request. Please upload a file."})
        file = request.files["file"]
        if file.filename == "":
            logging.warning("No file selected")
            return jsonify({"error": "No file selected. Please select a file."})
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                logging.info(f"Saving file to: {filepath}")
                file.save(filepath)
                if model is None:
                    logging.error("Model not loaded")
                    return jsonify({"error": "Model not loaded. Please check if 'emotion_model.pkl' exists and is valid."})
                emotion = predict_emotion(filepath)
                logging.info(f"Emotion predicted: {emotion}")
                return jsonify({
                    "message": "File processed successfully!",
                    "emotion": emotion,
                    "uploaded_file_path": url_for('uploaded_file', filename=filename)
                })
            except Exception as e:
                logging.error(f"Error processing file: {e}")
                logging.error(traceback.format_exc())
                return jsonify({"error": f"An error occurred: {str(e)}"})
        else:
            logging.warning("Invalid file type")
            return jsonify({"error": "Invalid file type. Please upload a WAV file."})
    return render_template("upload.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)