import os
import joblib
import soundfile
import numpy as np
import librosa
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"wav"}

# Load model and data once, to save processing time.
model = joblib.load("model.pkl")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0
            )
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    message = ""
    uploaded_file_path = None
    if request.method == "POST":
        # Check if a file is included in the POST request
        if "file" not in request.files:
            return "No file part in the request. Please upload a file.", 400

        file = request.files["file"]

        # Check if a file was selected
        if file.filename == "":
            return "No file selected. Please select a file.", 400

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            # Extract features and predict emotion
            feature = extract_feature(filepath, mfcc=True, chroma=True, mel=True)
            emotion = model.predict([feature])[0]
            message = f'File uploaded and processed successfully!'
            uploaded_file_path = url_for("uploaded_file", filename=filename)

    return render_template(
        "upload.html", emotion=emotion,message=message, uploaded_file_path=uploaded_file_path
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
