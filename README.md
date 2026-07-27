# ASL Model Detection

This project recognizes ASL (American Sign Language) alphabet letters from hand images.
Instead of training directly on image pixels, it uses MediaPipe hand landmarks,
builds geometric features, and trains a supervised classifier.

## Goal

Classify ASL letters using compact and interpretable hand-landmark features.

## Project Structure


pipeline.py                          Full project pipeline
src/extraction/                      MediaPipe extraction logic
src/preprocessing/                   Data checks and cleaning
src/training/                        Validation and model training
src/app/                             Webcam demo app
src/util/                            Utility helpers
models/                              Saved models and evaluation outputs
notebooks/dataset/                   Dataset setup notebook
notebooks/preprocessing/             EDA and preprocessing analysis notebooks
notebooks/modeling/                  Model selection and report figure notebooks
data/                                Dataset and generated CSV files


## Pipeline

1. **Extract landmarks**
   Read dataset images and extract 21 normalized hand landmarks with MediaPipe.

   Main outputs:

   
   data/asl_landmarks_mediapipe.csv
   data/asl_landmarks_failed.csv


2. **Run data quality checks**
   Remove invalid rows, missing landmarks, geometric outliers, and inconsistent hand structures.

3. **Create engineered features**
   Generate extra geometric features (finger distances, joint angles, wrist-to-fingertip vectors).

   Output:

   
   data/asl_features_engineered.csv
   

4. **Select best model**
   Compare multiple classifiers with nested cross-validation and choose the best one by macro F1 score.

   Outputs:

    
   models/group_nested_kfold_cv_results.csv
   models/group_nested_kfold_cv_fold_results.csv
    

5. **Final training**
   Retrain the selected model on the full processed dataset and save it for inference/demo use.

   Output:

    
   models/asl_model.pkl
    

## Setup

 powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
 

## Run

Extract landmarks only:

 powershell
.\.venv\Scripts\python.exe src\extraction\mediapipe_landmarks_extractor.py
 

Run the full pipeline:

 powershell
.\.venv\Scripts\python.exe pipeline.py
 

Start webcam demo:

 powershell
.\.venv\Scripts\python.exe src\app\sign_language_app.py
 

## Dataset

Put the dataset under:

 
data/ASL_Alphabet_Dataset/asl_alphabet_train/
 

Expected format: one subfolder per letter/class.

## Notes

- Generated CSV files in `data/` can be large and may not be committed.
- `models/hand_landmarker.task` is required for MediaPipe landmark extraction.
- Nested cross-validation can take a long time on the full dataset.

