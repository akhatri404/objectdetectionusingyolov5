import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

#!pip install streamlit opencv-python-headless torch torchvision numpy
#!pip install git+https://github.com/ultralytics/yolov5.git

#pip install opencv-python
#pip install torch
#pip install psutil
#pip install torchvision

st.set_page_config(page_title="object detector")
@st.cache_data # Cache the pipeline object to speed up processing
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5s.pt')

# Define a function for detecting objects in an image
def detect_objects(image):
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Perform object detection on the image
    results = model(image_np)

    # Extract the bounding boxes and class labels for each object detected
    boxes = results.xyxy[0][:, :4]
    labels = results.xyxy[0][:, 5]
    scores = results.xyxy[0][:, 4]

    # Get the class names for each label
    class_names = model.names

    # Create a list to store the labels with class names
    labeled_labels = []

    # Loop over the labels and add the class name to each one
    for label, score in zip(labels, scores):
        class_name = class_names[int(label)]
        labeled_label = f'{class_name} ({score:.2f})'
        labeled_labels.append(labeled_label)


    return boxes, labeled_labels

# Create a Streamlit app
st.title("Object Detection with YOLOv5")
st.write('**prepared by Er Ashish KC Khatri**')

# Add a file uploader to the app
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

# If a file is uploaded, display the image and detect objects
if uploaded_file is not None:
    # Load the image using Pillow
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect objects in the image
    boxes, labels = detect_objects(image)

     # Draw bounding boxes around the objects in the image
    image_np = np.array(image)
    for box, label in zip(boxes, labels):
        cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image_np, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with the bounding boxes
    st.image(image_np, caption='Detected Objects', use_column_width=True)
