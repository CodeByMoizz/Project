from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def dashboard():

    result = None
    recent_events = []
    error_message = None

    if request.method == "POST":

        audio = request.files.get("audio")

        if audio and audio.filename:

            error_message = "Test Error: Analyze button is working!"

    return render_template(
        "dashboard.html",
        result=result,
        recent_events=recent_events,
        error_message=error_message
    )


@app.route("/event-history")
def event_history():
    return render_template("event_history.html")


@app.route("/manual-review")
def manual_review():
    return render_template("manual_review.html")


@app.route("/reports")
def reports():
    return render_template("reports.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


if __name__ == "__main__":
    app.run(debug=True)