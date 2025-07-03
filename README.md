
# Speech Emotion Recognition 🎙️

This project focuses on recognizing human emotions from speech audio using machine learning techniques. It involves preprocessing, feature extraction, model training, and evaluation.

## 📌 Features
- Noise reduction using `noisereduce`
- Feature extraction from audio files using `librosa`
- Emotion classification with `RandomForestClassifier`
- Hyperparameter tuning using `GridSearchCV`
- Model evaluation with accuracy and confusion matrix

## 🗂️ Dataset
The dataset used is the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, containing labeled audio samples representing different emotional states.

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

### 3. Run the Notebook
Open the notebook with VS Code or Jupyter:
```bash
jupyter notebook Speech_emotion_recognition.ipynb
```

## 🧠 Model Details
- **Algorithm**: Random Forest
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report
- **Tuning**: GridSearchCV for optimal hyperparameters

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

## 📝 License
This project is licensed under the MIT License.

---

Feel free to fork, improve, and contribute!
