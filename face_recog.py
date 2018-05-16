#!/bin/env python
import cv2
import os
import numpy as np

subject = ["", "Nathan Worstell", "Scott Worstell"]
saved_model_path = 'saved_model/smart_doorbell_model.yaml'
test_data_path = 'test_data/'

def detect_face(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv_files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    
    return gray_image[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    faces = []
    labels = []

    print "Training model..."

    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            cv2.imshow("training_data", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

print("Preparing data...\n")
faces, labels = prepare_training_data("training_data")
print("\nData prepared\n")

face_recognizer = cv2.face.createLBPHFaceRecognizer()
face_recognizer.train(faces, np.array(labels))
face_recognizer.save(saved_model_path)
# to load the model
# face_recognizer.load(saved_model_path)

def draw_rectangle(image, rect):
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_image, img):
    image = test_image.copy()
    face, rect = detect_face(image)
    label, confidence = face_recognizer.predict(face)
    label_text = subject[label]
    draw_rectangle(image, rect)
    draw_text(image, label_text, rect[0], rect[1]-5)

    print "{}% confident {} is in {}".format(round(confidence, 3), label_text, img)
    
    return image

print("Predicting images...\n")

test_dir = os.listdir(test_data_path)
count = 1

for img in test_dir:
    cv2.imshow("Subject {}".format(count), cv2.resize(predict(cv2.imread("test_data/" + img), str(img)), (400, 500)))
    count += 1

cv2.waitKey(0)