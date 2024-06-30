from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import os

app = Flask(__name__)

model = load_model('./models/best_fit_model.keras')

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)
        return np.expand_dims(mfccs, axis=0)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        features = extract_features(file_path)
        if features is None:
            return jsonify({"error": "Error in feature extraction"}), 500
        prediction = model.predict(features)
        print(f"Prediction scores: {prediction}") 
        predicted_emotion = np.argmax(prediction, axis=1)
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps']
        result = emotions[predicted_emotion[0]]
        return jsonify({'emotion': result})

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
