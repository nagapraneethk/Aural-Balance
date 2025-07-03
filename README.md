
# Aural Balance 🎙️

This project focuses on recognizing human emotions from speech audio using machine learning techniques. It involves preprocessing, feature extraction, model training, and evaluation.

## 📌 Features
- Noise reduction using `noisereduce`
- Feature extraction from audio files using `librosa`
- Emotion classification with `RandomForestClassifier`
- Hyperparameter tuning using `GridSearchCV`
- Model evaluation with accuracy and confusion matrix

## 🗂️ Datasets

This project primarily uses the **RAVDESS** dataset, but is compatible with other speech emotion recognition datasets:

### Primary Dataset
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
  - Contains 7,356 files with 24 professional actors
  - 8 emotions: calm, happy, sad, angry, fearful, surprise, disgust, neutral
  - Links: 
    - [RAVDESS Original](https://zenodo.org/record/1188976)
    - [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

### Alternative Compatible Datasets
- **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
  - 7,442 clips from 91 actors
  - 6 emotions: anger, disgust, fear, happy, neutral, sad
  - Links:
    - [CREMA-D Original](https://github.com/CheyneyComputerScience/CREMA-D)
    - [CREMA-D on Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad)

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/nagapraneethk/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Navigate to Notebooks Directory
```bash
cd notebooks
```

### 4. Run the Notebook
Open the notebook with VS Code or Jupyter:
```bash
jupyter notebook Speech_emotion_recognition.ipynb
```

## 🧠 Model Details
- **Algorithm**: Random Forest
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report
- **Tuning**: GridSearchCV for optimal hyperparameters
- **Model Storage**: Trained models are saved in the `models/` directory

## 📁 Project Structure
```
AuralBalance/
├── notebooks/
│   ├── Speech_emotion_recognition.ipynb    # Main training notebook
│   ├── data_exploration_notebook.ipynb     # Data analysis and visualization
│   └── training_model_notebook.ipynb       # Alternative training approach
├── models/
│   ├── best_rf_model.joblib                # Best performing Random Forest model
│   ├── emotion_recognition_model.pkl       # Alternative trained model
│   ├── label_encoder.pkl                   # Label encoder for emotion mapping
│   └── svm_emotion_model.pkl               # SVM-based emotion model
├── emotion_predictor.py                    # Production inference script
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

## 📊 Results
- Confusion matrix displayed for multi-class emotion classification
- Achieved good accuracy with generalizable performance on unseen data

## 📦 Dependencies
- numpy
- librosa
- noisereduce
- scikit-learn
- matplotlib
- seaborn
- joblib
- sounddevice (for real-time audio recording)
- scipy (for audio I/O operations)

## 🎯 Usage Notes
- The model is trained on RAVDESS but can be adapted for other datasets
- Different datasets may require adjustment of emotion labels and feature extraction parameters
- For optimal performance, ensure audio files are preprocessed consistently across datasets
- **Notebooks**: All Jupyter notebooks are located in the `notebooks/` directory
- **Models**: Trained models and encoders are stored in the `models/` directory
- **Inference**: Use `emotion_predictor.py` for real-time emotion prediction

## 🚀 Quick Start for Inference
```python
from emotion_predictor import predict_emotion

# Predict emotion from an audio file
emotion = predict_emotion('path/to/your/audio.wav')
print(f"Predicted emotion: {emotion}")
```

## 📝 License
This project is licensed under the MIT License.

---

Feel free to fork, improve, and contribute!
