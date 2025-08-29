import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

# ----- Load model -----
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Pass Sequential as custom object
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
model.load_weights("facialemotionmodel.h5")


# Labels (adjust to your training dataset)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Haarcascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# ----- Open webcam -----
cap = cv2.VideoCapture(0)   # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess face for model
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # add channel
        roi_gray = np.expand_dims(roi_gray, axis=0)   # batch dimension

        # Predict emotion
        preds = model.predict(roi_gray)
        label = emotions[np.argmax(preds)]

        # Put text above face
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video
    cv2.imshow('Facial Emotion Recognition', frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
