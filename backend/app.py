import os
import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image, ImageChops
from flask_cors import CORS

app = Flask(__name__)  

CORS(app)  # Allow frontend access

# Load the trained DenseNet model
MODEL_PATH = "model/densenet_image_forgery_detection.h5"
model = load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to apply Error Level Analysis (ELA)
def ELA(img, quality=80, threshold=50):
    TEMP = 'temp_ela.jpg'
    SCALE = 10

    img = img.convert("RGB")  # Ensure image is in RGB format
    img.save(TEMP, "JPEG", quality=quality)
    temporary = Image.open(TEMP)
    diff = ImageChops.difference(img, temporary)

    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            r, g, b = d[x, y]
            modified_intensity = int(0.2989 * r + 0.587 * g + 0.114 * b)
            d[x, y] = (modified_intensity * SCALE,) * 3

    binary_mask = diff.point(lambda p: 255 if p >= threshold else 0)
    return binary_mask

# Function to preprocess ELA image
def preprocess_image(image):
    ela_image = ELA(image)
    ela_image = ela_image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(ela_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to generate Grad-CAM heatmap for DenseNet
def generate_gradcam(image, model, layer_name="conv5_block16_concat"):
    img_array = preprocess_image(image)
    
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

    return heatmap

# Function to overlay heatmap on the input image (not ELA)
def overlay_heatmap(image, heatmap, alpha=0.5):
    # Convert PIL image to OpenCV format (RGB -> BGR)
    original_image = np.array(image.convert("RGB"))
    original_image = cv2.resize(original_image, (224, 224))  # Resize to match model input

    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (224, 224))

    # Convert heatmap to color
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color mapping
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

    # Convert overlay to base64 for frontend display
    overlay_img = Image.fromarray(overlay)
    buffered = BytesIO()
    overlay_img.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_string

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(file.stream)

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0][0]

    result = "Forged" if prediction > 0.5 else "Original"

    response_data = {"result": result, "confidence": float(prediction)}

    # If forged, generate Grad-CAM heatmap and ELA image
    if result == "Forged":
        # Generate ELA image
        ela_image = ELA(image)
        ela_image = ela_image.resize((224, 224))
        buffered = BytesIO()
        ela_image.save(buffered, format="JPEG")
        ela_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        response_data["ela_image"] = ela_encoded  # Only return for forged images

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam(image, model)
        heatmap_overlay = overlay_heatmap(image, heatmap)  # Overlay on input image
        response_data["heatmap"] = heatmap_overlay  # Send heatmap in response

    return jsonify(response_data)

if __name__ == "__main__":  
    app.run(debug=True, port=5000)