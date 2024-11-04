# -*- coding: utf-8 -*-
"""Pose_Detection_alert.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TRr9t6ATVl7CQ9_0zRTBGmv3KA8gJiqp
"""

!pip install mediapipe --upgrade

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import load_model
import smtplib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('icml_face_data.csv')  # Adjust path as necessary

# Preprocess the data
X = []
y = []

for index, row in data.iterrows():
    # Convert pixel string to numpy array
    pixels = np.fromstring(row[2], sep=' ').reshape(48, 48).astype('float32')
    X.append(pixels)
    y.append(row['emotion'])  # Assuming 'emotion' column has labels

X = np.array(X)
X = np.expand_dims(X, axis=-1)  # Add channel dimension
X /= 255.0  # Normalize pixel values

# Encode labels (e.g., anger=0, annoyance=1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for number of classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Save the model
model.save('facial_expression_model.h5')

model = load_model('facial_expression_model.h5')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

#  send alert via email
def send_alert():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("sender_account@gmail.com", "password")

    msg ="Alert! Fight Detected!"
    server.sendmail("sender_account@gmail.com", "reciver_account@gmail.com", msg)

    server.quit()

# Function to predict facial expression
def predict_expression(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)

    prediction = model.predict(face_image)
    return np.argmax(prediction)  # Returne predicted class index

# Function to extract face region (implement based on  needs)
def extract_face_region(frame):
    # This is a placeholder.....can use a face detection method (like Haar cascades or dlib) to get the face region
    #  using a fixed region
    height, width, _ = frame.shape
    return frame[int(height/4):int(height/4) + 48, int(width/4):int(width/4) + 48]  # Example fixed region

# Start video capture
cap = cv2.VideoCapture(0)
count = 0
while count<50:
    ret, frame = cap.read()

    if not ret:
      print('broken here')
      break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)


    if results.pose_landmarks:
        # Draw landmarks on frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Example logic to check for fight position....should be customized
        #  simply check if certain landmarks are in specific positions
        left_hand_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_hand_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y

        if left_hand_y < 0.5 and right_hand_y < 0.5:  #  fight stance  #error
            print("Fight position detected!")


            #  face region n predict expression
            face_region = extract_face_region(frame)
            expression_index = predict_expression(face_region)

            #  'angry' == 0
            if expression_index == 0:
                print("Fight detected! Sending alert...")
                count+=1
                print(count)
            else:
              print('But no fight detected')
              print("====================================================")


    plt.imshow(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if count>0:
  send_alert()
  print("alert send succesfully")

cap.release()
cv2.destroyAllWindows()