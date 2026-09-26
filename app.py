from flask import Flask, render_template, request
from flask import abort

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
    statistics = {
        "total_events": 0,
        "critical_events": 0,
        "average_confidence": "N/A",
        "model_disagreements": 0,
        "poor_quality": 0,
    }
    return render_template(
        "reports.html",
        statistics=statistics,
        category_stats=[],
        report_data=[],
    )

@app.get("/export/<format>")
def export_report(format):
    if format not in {"csv", "excel"}:
        abort(404)

    abort(501, description="Report export is not implemented yet.")

@app.route("/admin")
def admin():
    admin_stats = {
        "total_events": 0,
        "critical_alerts": 0,
        "average_confidence": "N/A",
        "model_disagreements": 0,
        "poor_quality": 0,
    }
    settings = {
        "confidence_threshold": 0.70,
        "top_two_margin": 0.10,
    }

    return render_template(
        "admin.html",
        admin_stats=admin_stats,
        settings=settings,
    )


if __name__ == "__main__":
    app.run(debug=True)