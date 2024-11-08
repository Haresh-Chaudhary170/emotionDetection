import cv2
import numpy as np
import dlib
from keras.models import model_from_json

# Function to load the Keras model
def load_emotion_model(model_json_path, model_weights_path):
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights(model_weights_path)
    print("Loaded model from disk")
    return emotion_model

# Define emotions dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion detection model
emotion_model = load_emotion_model("emotion_model.json", "emotion_model.h5")

# Initialize face detector using dlib HOG
face_detector = dlib.get_frontal_face_detector()

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        # Detect faces using HOG
        num_faces = face_detector(gray_frame)

        for face in num_faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    except Exception as e:
        # Print the error if any
        print("Error:", str(e))
        pass

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
