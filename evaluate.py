from pathlib import Path
import csv
import joblib

from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score


EVAL_PATH = Path("eval_set")
LABELS_PATH = Path("eval_labels.csv")


# Load trained model
model = joblib.load("model.joblib")


images = []
true_labels = []
filenames = []


# Read true labels from CSV
with open(LABELS_PATH, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        filenames.append(row["ï»¿filename"])
        true_labels.append(row["true_label"])


# Predict each evaluation image
predicted_labels = []

for filename in filenames:

    image_path = EVAL_PATH / filename

    image = Image.open(image_path).convert("RGB")

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )

    prediction = model.predict([features])[0]

    predicted_labels.append(prediction)


# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

print("Evaluation images:", len(filenames))
print("Evaluation accuracy:", accuracy)

# Show a few predictions
print("\nSample predictions:")

for i in range(10):
    print(
        filenames[i],
        "→ Predicted:",
        predicted_labels[i],
        "| Actual:",
        true_labels[i]
    )