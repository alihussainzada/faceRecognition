import cv2
import numpy as np
from PIL import Image

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        face_region = gray_image[y:y + h, x:x + w]  # Extract the face region
        id, confidence = clf.predict(face_region)
        confidence_percentage = int(100 * (1 - confidence / 300))

        if confidence_percentage > 77:
            if id == 1:
                cv2.putText(img, "Ali", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            # Add more IDs and names as needed
        else:
            cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return img

def recognize(img, clf, faceCascade):
    img = draw_boundary(img, faceCascade, 1.3, 5, (255, 255, 255), "Face", clf)
    return img

def main():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        if not ret:
            print("Error: Unable to capture video")
            break

        img = recognize(img, clf, faceCascade)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
            break

    video_capture.release()
    cv2.destroyAllWindows()

main()
