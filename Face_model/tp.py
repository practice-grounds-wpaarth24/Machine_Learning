import cv2
import time
from deepface import DeepFace
import os


# Path to the reference image (the one you want to compare against)
reference_image_path = "C:/Users/Paarth/Desktop/VCET/Sahil_photo.jpeg"

# Open a connection to the default camera
cap = cv2.VideoCapture(0)

# Capture only one frame
ret, frame = cap.read()

if ret:
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Haar Cascades to detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Get the first face coordinates

        # Add a box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the captured frame with the box
        cv2.imshow('Captured Frame', frame)

        # Scanning animation (simulating a scanning effect with a moving line)
        for i in range(y, y + h):
            # Draw a scanning line
            scan_frame = frame.copy()
            cv2.line(scan_frame, (x, i), (x + w, i), (0, 255, 0), 2)
            cv2.imshow('Captured Frame', scan_frame)
            cv2.waitKey(10)  # Wait to create animation effect

        # Extract face after scanning animation
        face = frame[y:y+h, x:x+w]

        # Save the detected face temporarily (DeepFace needs image path)
        face_image_path = "scanned_face.jpg"
        cv2.imwrite(face_image_path, face)

        # Display the scanned face (optional)
        cv2.imshow('Scanned Face', face)
        cv2.waitKey(500)  # Display the scanned face for 500 ms

        # Compare the scanned face with the reference image using DeepFace
        try:
            # Use DeepFace for face verification (GPU is used if available)
            result = DeepFace.verify(img1_path=face_image_path, img2_path=reference_image_path)

            # Check if the face matches
            if result["verified"]:
                print("Face Matched!")
            else:
                print("No Match Found")

        except Exception as e:
            print(f"Error during verification: {str(e)}")

    else:
        print("No face detected in the frame")

cap.release()
cv2.destroyAllWindows()

os.remove(face_image_path) # Remove the temporary face image