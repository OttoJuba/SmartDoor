#!/bin/env python
import cv2
import os
import numpy as np

subject = ["", "subject2", "subject1"]
saved_model_path = 'saved_model/smart_doorbell_model.yaml'
test_data_path = 'test_data/'
face_recognizer = cv2.face.createLBPHFaceRecognizer()

def detect_face(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv_files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
    
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    
    return gray_image[y:y+w, x:x+h], faces[0]


def prepare_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    faces = []
    labels = []

    print "Data preparation phase started\n"

    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("subject_", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path) 
            
            cv2.imshow(subject_dir_path, cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)

    print "Data preparation phase complete\n"

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

def draw_rectangle(image, rect):
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_image, img):
    global face_recognizer

    image = test_image.copy()
    face, rect = detect_face(image)
    label, confidence = face_recognizer.predict(face)
    label_text = subject[label]
    draw_rectangle(image, rect)
    draw_text(image, label_text, rect[0], rect[1]-5)

    print "{}% confident {} is in {}".format(round(confidence, 3), label_text, img)
    
    return image


def main():
    
    faces, labels = prepare_data("training_data")
    
    face_recognizer.train(faces, np.array(labels))
    print "Model training complete\n"

    face_recognizer.save(saved_model_path)
    print "Model saved as: " + saved_model_path + "\n"
    
    # to load the model use the code below
    # face_recognizer.load(saved_model_path)

    print "Subject prediction phase started\n"

    test_dir = os.listdir(test_data_path)
    
    count = 1

    for img in test_dir:
        cv2.imshow("Subject {}".format(count), cv2.resize(predict(cv2.imread("test_data/" + img), str(img)), (400, 500)))
        count += 1

    print "\nSubject predicition phase complete"
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
    

