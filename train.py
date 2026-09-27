from pathlib import Path

from PIL import Image
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib


DATASET_PATH = Path("candidate_tiles")

images = []
labels = []

for class_folder in DATASET_PATH.iterdir():

    if not class_folder.is_dir():
        continue

    label = class_folder.name

    for image_path in class_folder.glob("*.png"):

        image = Image.open(image_path).convert("RGB")

        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=-1
        )

        images.append(features)
        labels.append(label)


print("Total images:", len(images))
print("Classes:", sorted(set(labels)))


X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


model = SVC(
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Validation accuracy:", accuracy)


joblib.dump(model, "model.joblib")

print("Model saved successfully.")