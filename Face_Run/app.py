from flask import Flask, render_template, request, jsonify
import cv2
import os
import time
from deepface import DeepFace

app = Flask(__name__)

# Path to the reference image
reference_image_path = "C:/Users/Paarth/Desktop/Resume/Resume/photo.jpeg"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan_face', methods=['POST'])
def scan_face():
    # Open a connection to the default camera
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # Allow the camera to warm up

    # Capture one frame
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            enlarge_by = 30
            x = max(0, x - enlarge_by)
            y = max(0, y - enlarge_by)
            w = w + 2 * enlarge_by
            h = h + 2 * enlarge_by

            # Ensure rectangle does not go outside the frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)

            # Extract face
            face = frame[y:y+h, x:x+w]

            # Save the detected face temporarily
            face_image_path = "scanned_face.jpg"
            cv2.imwrite(face_image_path, face)

            # Compare the scanned face with the reference image using DeepFace
            try:
                result = DeepFace.verify(img1_path=face_image_path, img2_path=reference_image_path)

                # Check if the face matches
                if result["verified"]:
                    message = "Face Matched!"
                else:
                    message = "No Match Found"
            except Exception as e:
                message = f"Error during verification: {str(e)}"

            return jsonify({"message": message})
        else:
            return jsonify({"message": "No face detected in the frame"})
    else:
        return jsonify({"message": "Failed to capture image"})

if __name__ == '__main__':
    app.run(debug=True)
