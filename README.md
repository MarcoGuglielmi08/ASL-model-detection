# 🤟 ASL Model Detection

An end-to-end Machine Learning pipeline for recognizing **American Sign Language (ASL)** alphabet letters from hand images using **MediaPipe Hand Landmarks** and engineered geometric features.

Instead of training directly on raw image pixels, this project extracts 21 hand landmarks, generates interpretable geometric features, and trains a supervised classifier for robust ASL letter recognition.

---

## ✨ Features

- 📷 Hand landmark extraction with **MediaPipe**
- 🧹 Automatic data integrity and quality checks
- 📐 Geometric feature engineering
- 🤖 Model selection using **Nested Cross-Validation**
- 📊 Statistical comparison of multiple classifiers
- 🔍 Explainability analysis using Permutation Importance and SHAP
- 🎥 Real-time webcam inference using the trained model

---

## 📁 Project Structure

```text
.
├── assets/                     MediaPipe model files
├── docs/                       Project documentation
├── models/                     Trained models
├── notebooks/
│   ├── modeling/               Modeling and evaluation notebooks
│   └── preprocessing/          EDA and preprocessing notebooks
├── results/
│   ├── error_analysis/         Error analysis outputs
│   ├── explainability/         SHAP and feature importance
│   ├── nested_cv/              Nested CV reports and grid search results
│   ├── preprocessing/          Preprocessing figures
│   └── statistical_tests/      Statistical test results
├── src/
│   ├── app/                    Webcam application
│   ├── extraction/             MediaPipe landmark extraction
│   ├── preprocessing/          Data cleaning and feature engineering
│   ├── training/               Model selection and training
│   └── util/                   Utility modules
├── pipeline.py                 Complete project pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline

### 1. Landmark Extraction

Extract the 21 normalized hand landmarks from every image using MediaPipe.

**Outputs**

```text
data/asl_landmarks_mediapipe.csv
data/asl_landmarks_failed.csv
```

---

### 2. Data Quality Checks

The extracted landmarks are validated through:

- Data integrity checks
- Missing landmark detection
- Geometric outlier detection
- Structural consistency verification

---

### 3. Feature Engineering

Generate additional geometric descriptors, including:

- Finger distances
- Joint angles
- Wrist-to-fingertip vectors

**Output**

```text
data/asl_features_engineered.csv
```

---

### 4. Model Selection

Compare multiple machine learning algorithms using **Nested Cross-Validation**.

The evaluation includes:

- Hyperparameter optimization
- Performance comparison
- Statistical significance testing
- Explainability analysis

**Outputs**

```text
results/nested_cv/
```

---

### 5. Final Training

Retrain the best-performing model on the entire processed dataset and export it for inference.

**Output**

```text
models/asl_model.pkl
```

---

## 🚀 Installation

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Usage

### Extract hand landmarks

```bash
python src/extraction/mediapipe_landmarks_extractor.py
```

### Run the complete pipeline

```bash
python pipeline.py
```

### Launch the webcam demo

```bash
python src/app/sign_language_app.py
```

---

## 📂 Dataset

The dataset is **not included** in this repository because of its size.

Download the dataset and place it inside:

```text
data/
└── ASL_Alphabet_Dataset/
    └── asl_alphabet_train/
```

The dataset must contain one subdirectory for each ASL alphabet class.

> **Note:** the repository already includes the final trained model (`models/asl_model.pkl`), so the webcam demo can be used without retraining the model.

---

## 📊 Generated Outputs

Running the pipeline automatically produces:

- Engineered feature datasets
- Nested Cross-Validation reports
- Hyperparameter search results
- Statistical test results
- Confusion matrices
- Explainability reports
- Final trained model

All evaluation artifacts are stored under the `results/` directory.

---

## 🛠️ Technologies

- Python
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- SHAP

---

## 📝 Notes

- Generated CSV files and analysis outputs may be large and are therefore not necessarily committed to the repository.
- `assets/hand_landmarker.task` is required for MediaPipe landmark extraction.
- Running the complete Nested Cross-Validation on the full dataset may require a significant amount of time.

---

## 📄 License

This project is released under the MIT License.