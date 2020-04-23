import os
import torch
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, send_from_directory
from Dog_Classification.human_detection import FaceDetection
from Dog_Classification.dog_detection import DogDetection
from Dog_Classification.dog_classification import DogClassification


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
use_cuda = torch.cuda.is_available()

face_detection = FaceDetection(APP_ROOT)
dog_detection = DogDetection(use_cuda)
dog_classifier = DogClassification(use_cuda, APP_ROOT)


@app.route("/", methods=["POST", "GET"])
def home_func():
    if request.method == 'POST':
        # all images will be uploaded to 'uploads' folder
        target = os.path.join(APP_ROOT, 'uploads/')

        # create image directory if not found
        if not os.path.isdir(target):
            os.mkdir(target)

        upload = request.files.getlist("file")[0]

        # save image
        filename = secure_filename(upload.filename)
        destination = "/".join([target, filename])
        upload.save(destination)

        if dog_detection.dog_detector(destination):  # if dog is detected
            first_text = "Hello, dog!"
            second_text = "I think you are a"

        elif face_detection.face_detector(destination):  # if human is detected
            first_text = "Hello, human!"
            second_text = "You look like a"

        else:  # if it could not recognize dog nor human
            first_text = "I don't recognize you!"
            second_text = "You appear to be similar to a"

        breed = dog_classifier.predict_breed(destination)
        return render_template("results.html", image_name=filename, breed=breed, first=first_text, second=second_text)

    else:
        return render_template("home.html")


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory("uploads", filename)


if __name__ == "__main__":
    app.run()
