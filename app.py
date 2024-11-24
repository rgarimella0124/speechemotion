import os
import pickle
import soundfile
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import traceback


def setup_logging(app, log_dir="logs"):
    """
    Configure application-wide logging with rotating file handlers and console output.

    Sets up three log files:
    - app.log: Contains INFO and above messages
    - error.log: Contains ERROR and above messages
    - debug.log: Contains DEBUG and above messages

    Args:
        app (Flask): Flask application instance
        log_dir (str): Directory where log files will be stored, defaults to "logs"

    Note:
        - Each log file is limited to 10MB with rotation
        - app.log and error.log keep 5 backups
        - debug.log keeps 3 backups
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure formatter with detailed information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Set up main application log file for general information
    app_log_file = os.path.join(log_dir, 'app.log')
    file_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Set up error log file for tracking critical issues
    error_log_file = os.path.join(log_dir, 'error.log')
    error_file_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    # Set up debug log file for detailed debugging information
    debug_log_file = os.path.join(log_dir, 'debug.log')
    debug_file_handler = RotatingFileHandler(
        debug_log_file,
        maxBytes=10485760,  # 10MB
        backupCount=3
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(formatter)

    # Set up console output for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Clean up existing handlers and configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)

    # Add all handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(debug_file_handler)
    root_logger.addHandler(console_handler)

    # Minimize Werkzeug logging noise
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    app.logger.info('Logging system initialized')


# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"wav"}

# Set up logging before any other operations
setup_logging(app)

# Load model with error handling
model = None
try:
    model_path = os.path.join(os.path.dirname(__file__), "emotion_model.pkl")
    app.logger.info(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        app.logger.error(f"Model file does not exist at: {model_path}")
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    app.logger.error(traceback.format_exc())


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): Name of the file to check

    Returns:
        bool: True if file extension is allowed, False otherwise

    Note:
        Allowed extensions are defined in app.config["ALLOWED_EXTENSIONS"]
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def extract_feature(file_name, mfcc, chroma, mel):
    """
    Extract audio features from a file with detailed logging of each step.

    Args:
        file_name (str): Path to the audio file
        mfcc (bool): Whether to extract MFCC features
        chroma (bool): Whether to extract chroma features
        mel (bool): Whether to extract mel spectrogram features

    Returns:
        numpy.ndarray: Concatenated features or None if extraction fails
    """
    try:
        app.logger.info(f"Starting feature extraction for file: {file_name}")
        app.logger.debug(
            f"Extraction parameters - MFCC: {mfcc}, Chroma: {chroma}, Mel: {mel}")

        # Load audio file
        app.logger.debug(f"Loading audio file: {file_name}")
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            app.logger.info(f"Audio file loaded successfully - Sample rate: {
                            sample_rate}Hz, Duration: {len(X)/sample_rate:.2f}s")
            app.logger.debug(f"Audio data shape: {
                             X.shape}, Data type: {X.dtype}")

        result = np.array([])

        # Extract MFCC features
        if mfcc:
            app.logger.debug("Extracting MFCC features...")
            try:
                mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
                mfccs_processed = np.mean(mfccs.T, axis=0)
                app.logger.debug(
                    f"MFCC features shape before averaging: {mfccs.shape}")
                app.logger.debug(f"MFCC features shape after averaging: {
                                 mfccs_processed.shape}")
                result = np.hstack((result, mfccs_processed))
                app.logger.info(
                    f"MFCC features extracted successfully - {len(mfccs_processed)} features")
            except Exception as e:
                app.logger.error(f"Error during MFCC extraction: {str(e)}")
                raise

        # Extract Chroma features
        if chroma:
            app.logger.debug("Extracting Chroma features...")
            try:
                stft = np.abs(librosa.stft(X))
                app.logger.debug(f"STFT shape: {stft.shape}")
                chroma_feat = librosa.feature.chroma_stft(
                    S=stft, sr=sample_rate)
                chroma_processed = np.mean(chroma_feat.T, axis=0)
                app.logger.debug(f"Chroma features shape before averaging: {
                                 chroma_feat.shape}")
                app.logger.debug(f"Chroma features shape after averaging: {
                                 chroma_processed.shape}")
                result = np.hstack((result, chroma_processed))
                app.logger.info(
                    f"Chroma features extracted successfully - {len(chroma_processed)} features")
            except Exception as e:
                app.logger.error(f"Error during Chroma extraction: {str(e)}")
                raise

        # Extract Mel features
        if mel:
            app.logger.debug("Extracting Mel spectrogram features...")
            try:
                mel_spec = librosa.feature.melspectrogram(y=X, sr=sample_rate)
                mel_processed = np.mean(mel_spec.T, axis=0)
                app.logger.debug(f"Mel spectrogram shape before averaging: {
                                 mel_spec.shape}")
                app.logger.debug(f"Mel spectrogram shape after averaging: {
                                 mel_processed.shape}")
                result = np.hstack((result, mel_processed))
                app.logger.info(
                    f"Mel spectrogram features extracted successfully - {len(mel_processed)} features")
            except Exception as e:
                app.logger.error(
                    f"Error during Mel spectrogram extraction: {str(e)}")
                raise

        app.logger.info(
            f"Feature extraction completed successfully - Total features: {result.shape[0]}")
        app.logger.debug(f"Final feature vector shape: {result.shape}")

        # Log feature statistics for debugging
        if len(result) > 0:
            app.logger.debug(f"Feature statistics - Mean: {result.mean():.4f}, Std: {result.std():.4f}, "
                             f"Min: {result.min():.4f}, Max: {result.max():.4f}")

        return result

    except Exception as e:
        app.logger.error(f"Error during feature extraction: {str(e)}")
        app.logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


def predict_emotion(file_path):
    """
    Predict the emotion from an audio file using the loaded model.

    Args:
        file_path (str): Path to the audio file to analyze

    Returns:
        str: Predicted emotion label

    Raises:
        ValueError: If model is not loaded or feature extraction fails
        Exception: For any other errors during prediction

    Notes:
        - Requires a pre-trained model loaded in the global 'model' variable
        - Extracts MFCC, Chroma, and Mel spectrogram features
        - Features are averaged across time before prediction
    """
    try:
        app.logger.info(f"Predicting emotion for file: {file_path}")

        # Verify model is loaded
        if model is None:
            app.logger.error("Attempting to predict with no model loaded")
            raise ValueError(
                "Model not loaded. Please check if 'emotion_model.pkl' exists and is valid.")

        # Extract features for prediction
        app.logger.debug("Starting feature extraction")
        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)

        # Verify feature extraction success
        if features is None:
            app.logger.error("Feature extraction failed")
            raise ValueError("Failed to extract features")

        # Reshape features for model input
        features = features.reshape(1, -1)
        app.logger.debug(f"Features reshaped for prediction. Shape: {
                         features.shape}")

        # Make prediction
        prediction = model.predict(features)
        app.logger.info(f"Prediction successful: {prediction[0]}")

        return prediction[0]

    except Exception as e:
        app.logger.error(f"Error predicting emotion: {e}")
        app.logger.error(traceback.format_exc())
        raise


@app.route("/")
def index():
    """
    Render the main upload page.

    Returns:
        str: Rendered HTML template
    """
    return render_template("upload.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    """
    Handle file upload and emotion prediction requests.

    Methods:
        GET: Return the upload form
        POST: Process the uploaded file and return prediction results

    Returns:
        For POST requests:
            JSON response containing:
            - message: Success message
            - emotion: Predicted emotion
            - uploaded_file_path: Path to the uploaded file
            - error: Error message if processing failed

        For GET requests:
            str: Rendered HTML template

    Notes:
        - Validates file presence and type
        - Securely saves file with sanitized filename
        - Performs emotion prediction on valid uploads
    """
    if request.method == "POST":
        app.logger.info("Received POST request for file upload")

        # Check if file was included in request
        if "file" not in request.files:
            app.logger.warning("No file part in the request")
            return jsonify({"error": "No file part in the request. Please upload a file."})

        file = request.files["file"]

        # Check if file was selected
        if file.filename == "":
            app.logger.warning("No file selected")
            return jsonify({"error": "No file selected. Please select a file."})

        # Process valid file
        if file and allowed_file(file.filename):
            try:
                # Secure the filename and save the file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                app.logger.info(f"Saving file to: {filepath}")
                file.save(filepath)

                # Verify model is loaded
                if model is None:
                    app.logger.error("Model not loaded")
                    return jsonify({"error": "Model not loaded. Please check if 'emotion_model.pkl' exists and is valid."})

                # Predict emotion
                emotion = predict_emotion(filepath)
                app.logger.info(f"Emotion predicted: {emotion}")

                # Return successful response
                return jsonify({
                    "message": "File processed successfully!",
                    "emotion": emotion,
                    "uploaded_file_path": url_for('uploaded_file', filename=filename)
                })

            except Exception as e:
                app.logger.error(f"Error processing file: {e}")
                app.logger.error(traceback.format_exc())
                return jsonify({"error": f"An error occurred: {str(e)}"})
        else:
            app.logger.warning("Invalid file type")
            return jsonify({"error": "Invalid file type. Please upload a WAV file."})

    return render_template("upload.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """
    Serve uploaded files from the upload directory.

    Args:
        filename (str): Name of the file to serve

    Returns:
        Response: File response from the upload directory

    Notes:
        - Uses Flask's secure_filename to prevent directory traversal attacks
        - Files are served from the configured UPLOAD_FOLDER
    """
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    # When running in debug mode, Flask will reload the application
    # and initialize logging twice, but that's okay
    app.run(debug=True)
