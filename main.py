import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 model

# Title of the web app
st.title("Object Detection App")
st.write("Upload a video or image to detect objects using the YOLO model.")
st.write("This app demonstrates YOLO object detection for both images and videos.")

# File uploader
uploaded_file = st.file_uploader("Choose a file (image or video)", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]  # Determine if the file is an image or video

    if file_type == "image":
        # Process the image
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Load the image
        image = cv2.imread(temp_image_path)

        # Run YOLO object detection
        results = model(image)

        # Annotate the image
        annotated_image = results[0].plot()

        # Save the annotated image
        output_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(output_image_path, annotated_image)

        # Display the annotated image
        st.image(output_image_path, caption="Processed Image with Detected Objects")

        # Display detected objects
        st.write("Objects detected:")
        detected_objects = set()
        for obj in results[0].boxes.data:
            detected_objects.add(model.names[int(obj[5])])
        for obj in detected_objects:
            st.write(f"- {obj}")

        # Clean up
        os.remove(temp_image_path)
        os.remove(output_image_path)

    elif file_type == "video":
        # Process the video
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Display the original video
        st.video(temp_video_path)
        st.write("Processing video...")

        # Create a temporary file for the output video
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Initialize video capture and writer
        video = cv2.VideoCapture(temp_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Process video frames
        detected_objects = set()
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Run YOLO object detection
            results = model(frame)

            # Annotate the frame
            annotated_frame = results[0].plot()

            # Collect detected objects
            for obj in results[0].boxes.data:
                detected_objects.add(model.names[int(obj[5])])

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        video.release()
        out.release()

        # Display the processed video
        st.video(output_video_path)

        # Display detected objects
        st.write("Objects detected in the video:")
        for obj in detected_objects:
            st.write(f"- {obj}")

        # Clean up
        os.remove(temp_video_path)
        os.remove(output_video_path)

st.write("Thank you for using the Object Detection App!")
