import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
import glob


# Extract features (mfcc, chroma, mel) from a sound file
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


# Emotions in the RAVDESS dataset
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Emotions to observe
observed_emotions = ["calm", "happy", "fearful", "disgust"]


# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    # Check if we have pre-processed data
    if os.path.isfile("data.pkl"):
        print("Loading data from file...")
        x, y = joblib.load("data.pkl")
    else:
        print("Processing data...")
        x, y = [], []
        files = glob.glob("./ravdess_data/Actor_*/*.wav")
        total_files = len(files)
        for file in tqdm(files, total=total_files, desc="Loading data"):
            file_name = os.path.basename(file)
            emotion = emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
        # Save the processed data
        joblib.dump((x, y), "data.pkl")
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Check if we have a pre-trained model
if os.path.isfile("model.pkl"):
    print("Loading model from file...")
    model = joblib.load("model.pkl")
else:
    print("Training model...")
    # Initialize the Multi Layer Perceptron Classifier
    model = MLPClassifier(
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate="adaptive",
        max_iter=500,
    )
    # Train the model
    model.fit(x_train, y_train)
    # Save the model to a file
    joblib.dump(model, "model.pkl")

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))


# Define a function to predict emotions from sound files
def predict_emotion(file_path):
    # Extract features from the sound file
    feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    # Reshape the feature to match the training data shape
    feature = feature.reshape(1, -1)
    # Make the prediction using the trained model
    emotion = model.predict(feature)[0]
    return emotion


# Get the list of all files in the "testingfiles" directory
testing_files = glob.glob("./testingfiles/*.wav")

# Run each file through the prediction function and print out the emotion
for file in testing_files:
    print(f"Predicting for {file}...")
    emotion = predict_emotion(file)
    print(f"Predicted Emotion: {emotion}")
