import os
import cv2
import numpy as np
import matplotlib.pyplot as plta
import seaborn as snsn
import plotly.express as pxp

# this for Loading Dataset and Analyze Composition
dataset_path = "dataset/"
image_files = os.listdir(dataset_path)

print(f"Total images found: {len(image_files)}")

# Categorize Images for Better Analysis
gesture_categories = {}  # Dictionary to store counts
for file in image_files:
    gesture_name = file.split("_")[0]  # Assuming filenames like "hello_1.jpg"
    gesture_categories[gesture_name] = gesture_categories.get(gesture_name, 0) + 1

# Visualizing the Dataset Composition
plta.figure(figsize=(10,5))
plta.bar(gesture_categories.keys(), gesture_categories.values(), color="skyblue")
plta.xlabel("Gesture Categories")
plta.ylabel("Number of Images")
plta.title("Dataset Composition - Indian Sign Language")
plta.xticks(rotation=45)
plta.show()

# Gesture Image
img = cv2.imread("dataset/b.jpg")  # Change filename if needed
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plta.imshow(img)
plta.title("Sample Gesture Image")
plta.axis("off")
plta.show()

#(Scatter Plot)
epochs = [1, 2, 3, 4, 5]
accuracy = [72, 78, 84, 88, 90]
plta.plot(epochs, accuracy, marker="o", linestyle="-", color="blue")
plta.xlabel("Epochs")
plta.ylabel("Accuracy (%)")
plta.title("Model Performance Over Training")
plta.grid(True)
plta.show()

# Pie Chart for Gesture Distribution
fig = pxp.pie(names=list(gesture_categories.keys()), values=list(gesture_categories.values()), title="Gesture Frequency")
fig.show()
