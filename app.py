import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"wav"}


# Define a function to check if a file has an allowed extension
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    message = ""
    if request.method == "POST":
        # Check if a file is included in the POST request
        if "file" not in request.files:
            return "No file part in the request. Please upload a file.", 400

        file = request.files["file"]

        # Check if a file was selected
        if file.filename == "":
            return "No file selected. Please select a file.", 400

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            message = f'File "{filename}" uploaded and processed successfully!'

    return render_template("upload.html", message=message)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
