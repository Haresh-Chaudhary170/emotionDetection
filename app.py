import cv2
import numpy as np
import dlib
from keras.models import model_from_json
from flask import Flask, request, jsonify, render_template, Response
import os

app = Flask(__name__)

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

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model files
model_json_path = os.path.join(script_dir, "emotion_model.json")
model_weights_path = os.path.join(script_dir, "emotion_model.h5")

# Load the emotion detection model
emotion_model = load_emotion_model(model_json_path, model_weights_path)

# Initialize face detector using dlib HOG
face_detector = dlib.get_frontal_face_detector()


def detect_emotion(image_path):
    try:
        # Load the image
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using HOG
        num_faces = face_detector(gray_frame)

        emotions = []

        for face in num_faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotions.append(emotion_dict[maxindex])

        return emotions

    except Exception as e:
        # Print the error if any
        print("Error:", str(e))
        return []
def generate_frames():
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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/live_emotion_detection')
def live_emotion_detection():
    return render_template('live_emotion_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route to serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def route_detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    image_path = 'temp_image.jpg'  # Save the image temporarily

    image.save(image_path)

    detected_emotions = detect_emotion(image_path)

    return jsonify({'emotions': detected_emotions})

if __name__ == '__main__':
    app.run(debug=True)




















# # In your app.py, load the emotion detection model
# from keras.models import model_from_json
# # In your app.py, import OpenCV and NumPy
# import cv2
# import numpy as np
# from flask import Flask, request, jsonify, render_template

# app = Flask(__name__)

# # Load the emotion detection model architecture from JSON
# with open('emotion_model.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
# emotion_model = model_from_json(loaded_model_json)

# # Load the emotion detection model weights
# emotion_model.load_weights('emotion_model.h5')


# # ... (your other code)

# @app.route('/realtime_emotion', methods=['GET'])
# def realtime_emotion():
#     return render_template('index.html')

# # ... (other routes)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
