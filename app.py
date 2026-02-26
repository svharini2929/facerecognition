from flask import Flask, render_template, request
from deepface import DeepFace
import os

app = Flask(__name__)

KNOWN_FACES_DIR = "known_face"

@app.route("/", methods=["GET", "POST"])
def index():
    name = "No face detected"

    if request.method == "POST":
        file = request.files["image"]

        if file:
            # Save uploaded image temporarily
            test_image_path = "temp.jpg"
            file.save(test_image_path)

            match_found = False

            for known_file in os.listdir(KNOWN_FACES_DIR):
                known_image_path = os.path.join(KNOWN_FACES_DIR, known_file)

                try:
                    result = DeepFace.verify(
                        known_image_path,
                        test_image_path,
                        enforce_detection=False
                    )

                    if result["verified"]:
                        name = os.path.splitext(known_file)[0]
                        match_found = True
                        break

                except:
                    pass

            if not match_found:
                name = "Unknown Person"

    return render_template("index.html", name=name)

if __name__ == "__main__":
    app.run(debug=True)