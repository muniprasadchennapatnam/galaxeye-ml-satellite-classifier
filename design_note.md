# GalaxEye ML Systems Take-Home Assignment

## Design Note

### 1. Overall Approach

The system is designed as a small offline service for classifying satellite image tiles by land-use type.

The main components are:

1. **Training pipeline** — reads labelled satellite images from the provided dataset and trains a classifier.
2. **Feature extraction** — converts each RGB image into HOG (Histogram of Oriented Gradients) features.
3. **Classifier** — uses an SVM classifier to predict the land-use class.
4. **FastAPI service** — provides a `/predict` endpoint that accepts one image tile.
5. **SQLite database** — stores the prediction result, confidence, filename, and timestamp.
6. **Evaluation script** — evaluates the trained model against the provided evaluation set and labels.

The complete prediction flow is:

```text
Satellite Image Tile
        |
        v
FastAPI /predict
        |
        v
Image preprocessing
        |
        v
HOG feature extraction
        |
        v
SVM classifier
        |
        v
Prediction + confidence
        |
        v
SQLite database
        |
        v
JSON response
```

The complete system runs locally and does not depend on a hosted model API or internet connection.

### 2. Model and Training

The provided training images are organised into class folders. The folder name is used as the label.

For example:

```text
candidate_tiles/
    Forest/
    River/
    Highway/
    Residential/
    Industrial/
    AnnualCrop/
    SeaLake/
```

The training script reads the images, converts them to RGB, extracts HOG features, and trains an SVM classifier.

The trained model is saved as:

```text
model.joblib
```

This means the API does not need to retrain the model whenever a prediction request arrives.

The training dataset contained 1050 images across 7 classes.

A validation split was used during training. The validation accuracy was approximately 78.1%.

The provided evaluation set contained 210 images. Evaluation against the provided labels produced approximately 76.2% accuracy.

Accuracy is treated as a model evaluation result rather than the main goal of this take-home exercise. The assignment focuses on the reasoning, system design, and working implementation.

### 3. Prediction Endpoint

The service exposes:

```text
POST /predict
```

The endpoint accepts one image tile.

For every request, the service:

1. Reads the uploaded image.
2. Converts it to RGB.
3. Extracts HOG features using the same configuration used during training.
4. Sends the features to the trained SVM model.
5. Gets the predicted class.
6. Gets the highest class probability as the confidence value.
7. Stores the result in SQLite.
8. Returns the result as JSON.

Example response:

```json
{
    "filename": "tile_004.png",
    "prediction": "Forest",
    "confidence": 0.9614
}
```

### 4. Result Storage

SQLite is used for storing prediction results because the assignment requires an offline system and the current implementation is a small working slice.

The database contains a `predictions` table with:

```text
id
filename
prediction
confidence
created_at
```

For example:

```text
1 | tile_004.png | Forest | 0.9614 | 2026-09-27T...
```

This allows results to remain available after the API request finishes.

In a larger version of the system, an analyst-facing query endpoint could be added to filter results by prediction class, confidence range, filename, or time.

### 5. Handling Low-Confidence Predictions

The current implementation returns the model confidence with every prediction instead of treating every prediction as equally reliable.

For example:

```text
Forest -> 0.96
```

is different from:

```text
Forest -> 0.42
```

A future production version could define a confidence threshold. Predictions below the threshold could be marked for manual review instead of being treated as reliable classifications.

The threshold should be selected using validation data and the cost of incorrect classifications rather than choosing an arbitrary value.

### 6. Assumptions

The main assumptions are:

* Input images are valid satellite image tiles.
* The images have the same general format as the provided dataset.
* The seven training classes represent the classes expected by the service.
* The model and application run on the same offline machine.
* The current SQLite storage is sufficient for this small working slice.
* The model is loaded when the application starts and reused for subsequent requests.

### 7. Questions I Would Ask

Before building a production version, I would clarify:

1. What image formats and image dimensions should the API officially support?
2. What should happen when an image does not belong to one of the seven known classes?
3. What confidence threshold should trigger manual review?
4. How many prediction requests are expected per day?
5. Does the analyst need filtering and search capabilities?
6. How long should prediction results be retained?
7. What monitoring or health checks are expected for the offline deployment?

### 8. Future Improvements

The current implementation intentionally focuses on the required thin slice.

Possible future improvements include:

* Better model selection and feature engineering.
* More detailed evaluation metrics such as per-class precision, recall, and confusion matrix.
* A confidence threshold and review workflow.
* An endpoint for querying stored predictions.
* Input validation and better error handling.
* Structured logging.
* Automated health checks.
* Database indexing for larger datasets.
* Model versioning so predictions can be associated with the exact model used.
