import cv2
import time
from deepface import DeepFace
import os


reference_image_path = "C:/Users/Paarth/Desktop/VCET/Sahil_photo.jpeg"

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0] 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Captured Frame', frame)

        for i in range(y, y + h):
            scan_frame = frame.copy()
            cv2.line(scan_frame, (x, i), (x + w, i), (0, 255, 0), 2)
            cv2.imshow('Captured Frame', scan_frame)
            cv2.waitKey(10)  
        face = frame[y:y+h, x:x+w]

        face_image_path = "scanned_face.jpg"
        cv2.imwrite(face_image_path, face)

        cv2.imshow('Scanned Face', face)
        cv2.waitKey(500) 
        try:
            result = DeepFace.verify(img1_path=face_image_path, img2_path=reference_image_path)

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

os.remove(face_image_path) 
