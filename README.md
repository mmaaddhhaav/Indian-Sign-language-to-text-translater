Indian Sign Language to Text Translator
An AI-powered accessibility solution that translates Indian Sign Language (ISL) gestures into readable English text in real time, bridging communication gaps for deaf and non-speaking individuals.

📌 Overview
This project implements a real-time computer vision system that recognizes Indian Sign Language gestures from live video input and converts them into readable text. Designed with accessibility as the core goal, it aims to reduce communication barriers between the deaf and non-speaking community and the general public.

The system uses hand landmark detection combined with a deep learning architecture to learn both the spatial and temporal patterns of gestures across sequential video frames.

📂 Dataset
Located in the dataset/ directory.

Contains labeled image samples representing multiple ISL gesture categories.

Used for training, validation, and evaluation of the gesture recognition model.

📊 Visualizations
The project includes exploratory data analysis and model performance visualizations to provide insight into both the dataset and the training process:

Visualization	Description
Bar Chart	Number of images per gesture category, highlighting dataset balance.
Scatter Plot	Model accuracy across training epochs, illustrating learning progression.
Pie Chart	Distribution of gesture categories within the dataset.
⚙️ Getting Started
Prerequisites
Install the required dependencies:

bash
pip install matplotlib seaborn plotly opencv-python pandas numpy
Running the Visualizations
Execute the visualization scripts to generate the charts described above:

bash
python visualize.py
🛠️ Tech Stack
Category	Tools
Language	Python
Computer Vision	OpenCV, MediaPipe
Visualization	Matplotlib, Seaborn, Plotly
Data Handling	Pandas, NumPy
Deep Learning	TensorFlow (CNN–LSTM architecture)
🎯 Learning Outcomes
Practical experience working with image datasets and preprocessing pipelines.

Implementation of gesture recognition using machine learning and deep learning techniques.

Application of data visualization to derive insights into dataset quality and model performance.

Hands-on exposure to real-time computer vision and accessibility-focused engineering.

👨‍💻 Author
Madhav Sharma
B.Tech Computer Science & Engineering
Passionate about AI, Machine Learning, and building accessibility-driven solutions.

📄 License
This project is intended for educational and accessibility purposes. Feel free to reach out for collaboration or feedback.
