import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
dataset_path = 'leapGestRecog'

# Preprocessing the data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Using 20% of the data for validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),  # Adjust size depending on model requirements
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Set as training data

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Set as validation data

# Building the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Saving the model
model.save('hand_gesture_model.h5')

# Add real-time prediction logic here as needed.
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the image
    roi = frame[100:300, 100:300]  # Adjust this depending on where your hand is in the frame
    img = cv2.resize(roi, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    gesture_index = np.argmax(prediction)
    gesture_confidence = prediction[0, gesture_index]

    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, f'Gesture: {gesture_index} Conf: {gesture_confidence:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()