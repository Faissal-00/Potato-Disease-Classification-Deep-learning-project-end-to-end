from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from google.cloud import storage
import tensorflow as tf
import numpy as np

model = None
class_names = ["Early Blight", "Late Blight", "Healthy"]

BUCKET_NAME = "potato_tff"  # Replace with your GCP bucket name


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/potatoes.h5",
            "/tmp/potatoes.h5",
        )
        model = tf.keras.models.load_model("/tmp/potatoes.h5")

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        image = request.files["file"]
        image = np.array(
            Image.open(image).convert("RGB").resize((256, 256))  # Image resizing
        )
    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file"}), 400

    image = image / 255  # Normalize pixel values to the range [0, 1]
    img_array = tf.expand_dims(image, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return jsonify({"class": predicted_class, "confidence": confidence})
