from flask import Flask, render_template, request
import datetime

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        entry_content = request.form.get("content")
        entry_date = datetime.datetime.now().strftime("%Y-%m-%d")
        print(f"Received entry: {entry_content} on {entry_date}")
    entries_with_date = [
        ("Sample entry 1", "2023-10-01", "Oct 01"),
        ("Sample entry 2", "2023-10-02", "Oct 02"),
        ("Sample entry 3", "2023-10-03", "Oct 03"),
    ]
    return render_template("home.html", entries=entries_with_date)

    return app
