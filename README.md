# GalaxEye ML Systems Take-Home Assignment

## Overview

This project implements an offline satellite image classification service.

The system:

* Trains an SVM classifier using HOG image features.
* Accepts satellite image tiles through a FastAPI endpoint.
* Predicts the land-use class.
* Calculates a confidence value.
* Stores the prediction in SQLite.
* Returns the prediction as JSON.

## Project Structure

```text
galaxeye_assignment/
│
├── candidate_tiles/       # Training images
├── eval_set/              # Evaluation images
├── eval_labels.csv        # Ground-truth labels for evaluation
│
├── train.py               # Model training
├── evaluate.py            # Model evaluation
├── app.py                 # FastAPI application
├── model.joblib            # Trained SVM model
├── predictions.db          # SQLite prediction database
│
├── design_note.md         # System design
├── README.md              # Project documentation
└── part3_answers.txt      # Problem-solving answers
```

## Requirements

Python 3 is required.

Install the dependencies:

```bash
pip install pillow scikit-learn scikit-image joblib fastapi uvicorn python-multipart
```

## Training the Model

Run:

```bash
python train.py
```

The training script:

1. Reads images from `candidate_tiles`.
2. Uses the folder name as the class label.
3. Converts images to RGB.
4. Extracts HOG features.
5. Splits the dataset into training and validation data.
6. Trains an SVM classifier.
7. Saves the trained model as `model.joblib`.

Example result from the current run:

```text
Total images: 1050
Validation accuracy: 0.780952380952381
Model saved successfully.
```

## Evaluating the Model

Run:

```bash
python evaluate.py
```

The evaluation script:

1. Loads `model.joblib`.
2. Reads the filenames and true labels from `eval_labels.csv`.
3. Loads the corresponding images from `eval_set`.
4. Extracts HOG features.
5. Generates predictions.
6. Compares predictions with the true labels.

Current evaluation result:

```text
Evaluation images: 210
Evaluation accuracy: 0.7619047619047619
```

## Running the API

Start the FastAPI application:

```bash
python -m uvicorn app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Prediction Endpoint

### `POST /predict`

Upload one satellite image tile.

The endpoint:

```text
Image
  ↓
HOG features
  ↓
SVM model
  ↓
Prediction + confidence
  ↓
SQLite database
```

Example response:

```json
{
    "filename": "tile_004.png",
    "prediction": "Forest",
    "confidence": 0.9614
}
```

## Database

The application creates:

```text
predictions.db
```

The database contains a `predictions` table with:

```text
id
filename
prediction
confidence
created_at
```

Each successful `/predict` request stores one prediction record.

## Offline Operation

The system is designed to run locally.

The classifier is stored in `model.joblib`, and the API does not call an external model API or cloud service during prediction.

## Scope

This implementation focuses on the required working slice:

```text
one tile
   ↓
classifier
   ↓
stored result
```

A full analyst query interface, production monitoring, authentication, advanced error handling, and model version management are outside the scope of this take-home implementation.
