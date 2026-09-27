from fastapi import FastAPI, UploadFile, File
from PIL import Image
from skimage.feature import hog
import joblib
import sqlite3
from datetime import datetime


app = FastAPI()


# Load trained model once when the application starts
model = joblib.load("model.joblib")


# Create database and table
def create_database():

    connection = sqlite3.connect("predictions.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# Create database when application starts
create_database()


@app.get("/")
def home():
    return {
        "message": "GalaxEye ML service is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image = Image.open(file.file).convert("RGB")

    # Extract HOG features
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )

    # Predict class
    prediction = model.predict([features])[0]

    # Get prediction probabilities
    probabilities = model.predict_proba([features])[0]

    # Highest probability = confidence
    confidence = max(probabilities)

    # Current time
    created_at = datetime.now().isoformat()

    # Store result in database
    connection = sqlite3.connect("predictions.db")

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (filename, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        file.filename,
        prediction,
        float(confidence),
        created_at
    ))

    connection.commit()
    connection.close()

    # Return response
    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": round(float(confidence), 4)
    }