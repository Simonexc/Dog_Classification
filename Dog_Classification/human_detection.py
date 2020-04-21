import os
import cv2


class FaceDetection:
    def __init__(self, app_root):
        self.app_root = app_root
        self.model = os.path.join(self.app_root, 'static/models/haarcascade_frontalface_alt.xml')
        self.face_cascade = cv2.CascadeClassifier(self.model)

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0
