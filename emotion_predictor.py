import numpy as np
import librosa
import joblib
import os

def extract_features(file_path, max_pad_len=174):
    """
    Enhanced feature extraction with more robust feature generation
    and padding/truncation
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        
        # Extract multiple features
        stft = np.abs(librosa.stft(y))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Spectral features
        spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Pad or truncate features to a consistent length
        def pad_trunc_feature(feature, max_len):
            if feature.shape[1] > max_len:
                return feature[:, :max_len]
            pad_width = max_len - feature.shape[1]
            return np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Pad all features
        mfccs = pad_trunc_feature(mfccs, max_pad_len)
        spectral_cent = pad_trunc_feature(spectral_cent, max_pad_len)
        spectral_bandwidth = pad_trunc_feature(spectral_bandwidth, max_pad_len)
        spectral_rolloff = pad_trunc_feature(spectral_rolloff, max_pad_len)
        chroma = pad_trunc_feature(chroma, max_pad_len)
        tonnetz = pad_trunc_feature(tonnetz, max_pad_len)
        zcr = pad_trunc_feature(zcr, max_pad_len)
        
        # Flatten features
        features = np.concatenate([
            mfccs.ravel(),
            spectral_cent.ravel(),
            spectral_bandwidth.ravel(),
            spectral_rolloff.ravel(),
            chroma.ravel(),
            tonnetz.ravel(),
            zcr.ravel()
        ])
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_emotion(file_path, 
                   model_path='../notebooks/emotion_recognition_model.pkl', 
                   encoder_path='../notebooks/label_encoder.pkl'):
    """Predict emotion for a given audio file."""
    try:
        # Load the saved model and label encoder
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        
        # Extract features
        features = extract_features(file_path)
        
        if features is None:
            print("Feature extraction failed.")
            return None
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Predict
        prediction_encoded = model.predict(features)
        prediction = label_encoder.inverse_transform(prediction_encoded)
        
        return prediction[0]

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

def batch_predict(directory):
    """Predict emotions for multiple audio files in a directory."""
    results = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            try:
                prediction = predict_emotion(file_path)
                results[filename] = prediction
                print(f"{filename}: {prediction}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results



def main():
    # Single file prediction
    
    test_audio_path = '../data/03-01-08-02-02-02-04.wav'
    predicted_emotion = predict_emotion(test_audio_path)
    print(f"Predicted Emotion for {test_audio_path}: {predicted_emotion}")
    
    # Batch prediction

    # test_directory = '../data'
    # batch_results = batch_predict(test_directory)

if __name__ == "__main__":
    main()