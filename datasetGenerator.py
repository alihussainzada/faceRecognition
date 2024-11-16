import cv2
import os

folder_path = "images"

# Create the folder
try:
    os.mkdir(folder_path)
    print(f"Folder '{folder_path}' created successfully!")
except FileExistsError:
    print(f"Folder '{folder_path}' already exists.")

def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_crp(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:  # Check if no faces are detected
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return  # Exit the function if the video capture fails

    id = 1
    img_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        face = face_crp(frame)
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (250, 250))
            file_name_path = f"{folder_path}/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (70, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped face", face)
            if cv2.waitKey(1) == 13 or img_id == 200:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed...")

generate_dataset()
