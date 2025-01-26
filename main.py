import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Use a pre-trained YOLOv8 model

# Title of the web app
st.title("Object Detection")
st.write("Upload a video or image to detect objects using the YOLO model.")
st.write("Note:- This is just a prototype of my work during the internship.")

# Upload file
uploaded_file = st.file_uploader("Choose a file (video or image)", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]  # Determine if the file is a video or image

    if file_type == "video":
        # Save the uploaded video to a temporary file
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Load the video
        st.video(temp_video_path)
        st.write("Processing video...")

        # Create a temporary file for the output video
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Initialize a set to store detected object names
        detected_objects = set()

        # Process the video frame by frame
        video = cv2.VideoCapture(temp_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Run YOLO object detection
            results = model(frame)

            # Annotate the frame with bounding boxes
            annotated_frame = results[0].plot()

            # Extract detected object names
            for obj in results[0].boxes.data:
                detected_objects.add(model.names[int(obj[5])])

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        video.release()
        out.release()

        # Display the processed video
        st.video(output_video_path)

        # Display detected object names
        st.write("Objects detected in the video:")
        for obj in detected_objects:
            st.write(f"- {obj}")

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(output_video_path)

    elif file_type == "image":
        # Save the uploaded image to a temporary file
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Load the image
        image = cv2.imread(temp_image_path)

        # Run YOLO object detection
        results = model(image)

        # Annotate the image with bounding boxes
        annotated_image = results[0].plot()

        # Save the annotated image to a temporary file
        output_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(output_image_path, annotated_image)

        # Display the annotated image
        st.image(output_image_path, caption="Processed Image with Detected Objects")

        # Extract detected object names
        detected_objects = set()
        for obj in results[0].boxes.data:
            detected_objects.add(model.names[int(obj[5])])

        # Display detected object names
        st.write("Objects detected in the image:")
        for obj in detected_objects:
            st.write(f"- {obj}")

        # Clean up temporary files
        os.remove(temp_image_path)
        os.remove(output_image_path)

st.write("Thanks for using the object detection app!")
