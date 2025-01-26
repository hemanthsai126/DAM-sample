import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with your specific model if needed

# Streamlit app title
st.title("Object Detection with YOLO")
st.write("Upload an image to detect objects using YOLO.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    with open(temp_image_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Load the image
    image = cv2.imread(temp_image_path)

    # Perform object detection
    results = model(image)

    # Annotate the image with bounding boxes
    annotated_image = results[0].plot()

    # Convert the annotated image from BGR to RGB for displaying in Streamlit
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the original and annotated images
    st.image(annotated_image_rgb, caption="Processed Image with Detected Objects", use_column_width=True)

    # Extract and display detected object names
    st.write("Objects detected:")
    detected_objects = set()
    for obj in results[0].boxes.data:
        detected_objects.add(model.names[int(obj[5])])

    for obj in detected_objects:
        st.write(f"- {obj}")

    # Clean up the temporary file
    os.remove(temp_image_path)

st.write("Thank you for using the YOLO Object Detection App!")
