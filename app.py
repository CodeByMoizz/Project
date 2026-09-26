import csv
import io
import json
import os
import time
import uuid

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from config.config import (
    ACTIVE_PYTHON_MODEL,
    CLASSES,
    CRITICAL_CLASSES,
    EVENT_STATUSES,
    GTM_MODEL_URL,
    GTM_MODEL_VERSION,
    MAX_BATCH_FILES,
    MAX_UPLOAD_BYTES,
    ROLES,
    SECRET_KEY,
    SEGMENT_DURATION_SEC,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    display_name,
)
from database import database
from src import analysis, metrics, python_model, rules

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

database.init_db()


def current_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    return database.get_user(user_id)


def has_role(user, allowed_roles):
    if user is None:
        return False

    return user["role"] in allowed_roles


def gtm_ready():
    model_file = os.path.join("gtm_model", "model.json")
    return bool(GTM_MODEL_URL) or os.path.exists(model_file)


def gtm_info():
    source = GTM_MODEL_URL or "/gtm_model/"

    return {
        "ready": gtm_ready(),
        "source": source,
        "version": GTM_MODEL_VERSION,
        "message": (
            "The Teachable Machine model is loaded in the browser."
            if gtm_ready()
            else "No GTM model found. Export your Teachable Machine audio project as "
            "TensorFlow.js into gtm_model/, or set the GTM_MODEL_URL environment variable."
        ),
    }


def page_context():
    user = current_user()

    return {
        "user": user,
        "classes": CLASSES,
        "class_names": {name: display_name(name) for name in CLASSES},
        "severities": ["Informational", "Low", "Medium", "High", "Critical"],
        "qualities": ["Good", "Acceptable", "Poor", "Unusable"],
        "statuses": EVENT_STATUSES,
        "model_status": python_model.model_status(),
        "gtm": gtm_info(),
    }


def save_upload(uploaded_file):
    filename = secure_filename(uploaded_file.filename or "")

    if not filename:
        return None, "The file has no usable name."

    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return None, (
            f"'{extension or 'unknown'}' is not a supported format. "
            f"Use one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{extension}"
    stored_path = os.path.join(UPLOAD_DIR, stored_name)

    try:
        uploaded_file.save(stored_path)
    except Exception as error:
        return None, f"The file could not be saved: {error}"

    if os.path.getsize(stored_path) == 0:
        os.remove(stored_path)
        return None, "The uploaded file is empty."

    return stored_path, None


@app.route("/register", methods=["GET", "POST"])
def register():
    error_message = None

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        role = request.form.get("role") or "user"
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip()

        if len(username) < 3:
            error_message = "The username must be at least 3 characters."
        elif len(password) < 6:
            error_message = "The password must be at least 6 characters."
        elif role not in ROLES:
            error_message = "That role is not allowed."
        elif database.get_user_by_username(username):
            error_message = "That username is already taken."
        else:
            # The very first account becomes the administrator so the app can
            # be set up on a fresh database.
            if database.count_users() == 0:
                role = "admin"

            user_id = database.add_user(
                username, generate_password_hash(password), role, full_name, email
            )
            database.add_audit(user_id, "register", f"role={role}")
            session["user_id"] = user_id
            return redirect(url_for("dashboard"))

    context = page_context()
    return render_template(
        "register.html", error_message=error_message, roles=ROLES, **context
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error_message = None

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = database.get_user_by_username(username)

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["user_id"]
            database.add_audit(user["user_id"], "login", "successful login")
            return redirect(url_for("dashboard"))

        database.add_audit(None, "failed_login", f"username={username}")
        error_message = "The username or password is not correct."

    context = page_context()
    context["user"] = None
    return render_template("login.html", error_message=error_message, **context)


@app.route("/logout")
def logout():
    user = current_user()

    if user:
        database.add_audit(user["user_id"], "logout", "signed out")

    session.clear()
    return redirect(url_for("login"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    message = None

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip()
        database.update_user_profile(user["user_id"], full_name, email)
        database.add_audit(user["user_id"], "profile_update", "profile updated")
        message = "Your profile has been saved."
        user = database.get_user(user["user_id"])

    context = page_context()
    context["user"] = user
    return render_template("profile.html", message=message, **context)


@app.route("/", methods=["GET", "POST"])
def dashboard():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    results = []
    error_message = None
    batch_summary = None

    if request.method == "POST":
        uploaded_files = [f for f in request.files.getlist("audio") if f and f.filename]

        if not uploaded_files:
            error_message = "Choose an audio file first."
        elif len(uploaded_files) > 1 and not has_role(
            user, ["admin", "operator", "reviewer", "maintenance"]
        ):
            error_message = "Your role is not allowed to upload a batch of files."
        else:
            if len(uploaded_files) > MAX_BATCH_FILES:
                uploaded_files = uploaded_files[:MAX_BATCH_FILES]
                batch_summary = f"Only the first {MAX_BATCH_FILES} files were processed."

            failures = []

            for uploaded_file in uploaded_files:
                stored_path, save_error = save_upload(uploaded_file)

                if save_error:
                    failures.append(f"{uploaded_file.filename}: {save_error}")
                    continue

                started = time.time()

                try:
                    detections, analyse_error = analysis.analyse_file(
                        stored_path, uploaded_file.filename, user["user_id"]
                    )
                except Exception as error:
                    detections, analyse_error = [], f"The analysis failed: {error}"

                if analyse_error:
                    failures.append(f"{uploaded_file.filename}: {analyse_error}")
                    continue

                for detection in detections:
                    detection["processing_seconds"] = round(time.time() - started, 2)

                results.extend(detections)
                database.add_audit(
                    user["user_id"], "upload", f"file={uploaded_file.filename}"
                )

            if failures:
                error_message = " | ".join(failures)

            if len(uploaded_files) > 1 and results:
                batch_summary = (
                    f"{len(uploaded_files)} file(s) uploaded, "
                    f"{len(results)} segment(s) analysed."
                )

    context = page_context()
    context["user"] = user

    return render_template(
        "dashboard.html",
        results=results,
        result=results[0] if results else None,
        error_message=error_message,
        batch_summary=batch_summary,
        recent_events=database.get_recent_detections(8),
        active_alerts=database.get_alerts("active", 5),
        stats=database.get_statistics(),
        segment_duration=SEGMENT_DURATION_SEC,
        **context,
    )


@app.route("/detection/<int:detection_id>")
def detection_detail(detection_id):
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    row = database.get_detection(detection_id)

    if row is None:
        context = page_context()
        context["user"] = user
        return render_template(
            "detection.html", detection=None, error_message="That event does not exist.", **context
        ), 404

    detection = dict(row)
    detection["python_scores"] = load_scores(row["python_scores"])
    detection["gtm_scores"] = load_scores(row["gtm_scores"])
    detection["overlapping_classes"] = load_scores(row["overlapping_classes"]) or []

    context = page_context()
    context["user"] = user

    return render_template(
        "detection.html",
        detection=detection,
        reviews=database.get_reviews(detection_id),
        error_message=None,
        **context,
    )


def load_scores(raw):
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except ValueError:
        return {}


def ranked_scores(scores):
    if not isinstance(scores, dict):
        return []

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


app.jinja_env.globals["ranked_scores"] = ranked_scores
app.jinja_env.globals["display_name"] = display_name


@app.route("/audio/<int:audio_id>")
def audio_file(audio_id):
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    row = database.get_audio_file(audio_id)

    if row is None or not row["stored_path"] or not os.path.exists(row["stored_path"]):
        return "The audio file is no longer available.", 404

    return send_file(row["stored_path"])


@app.route("/live")
def live():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    context = page_context()
    context["user"] = user

    return render_template(
        "live.html",
        recent_events=database.get_recent_detections(8),
        active_alerts=database.get_alerts("active", 5),
        segment_duration=SEGMENT_DURATION_SEC,
        **context,
    )


# Receives one live microphone window, runs only the Python model on it, and
# returns the detection id so the browser can post its own GTM result next.
@app.route("/api/analyse-window", methods=["POST"])
def analyse_window():
    user = current_user()

    if user is None:
        return jsonify({"error": "You are not signed in."}), 401

    uploaded_file = request.files.get("audio")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No audio window was received."}), 400

    stored_path, save_error = save_upload(uploaded_file)

    if save_error:
        return jsonify({"error": save_error}), 400

    try:
        detections, analyse_error = analysis.analyse_file(
            stored_path, uploaded_file.filename, user["user_id"], source_kind="live"
        )
    except Exception as error:
        return jsonify({"error": f"The analysis failed: {error}"}), 500

    if analyse_error:
        return jsonify({"error": analyse_error}), 400

    if not detections:
        return jsonify({"error": "The window contained no usable audio."}), 400

    database.add_audit(user["user_id"], "live_window", f"audio_id={detections[0]['audio_id']}")

    return jsonify({"detections": [serialise_detection(d) for d in detections]})


def serialise_detection(detection):
    return {
        "detection_id": detection.get("detection_id"),
        "audio_id": detection.get("audio_id"),
        "segment_index": detection.get("segment_index"),
        "python_class": detection.get("python_class"),
        "python_class_name": display_name(detection.get("python_class") or "unknown"),
        "python_confidence": detection.get("python_confidence"),
        "python_top3": [
            [display_name(name), value] for name, value in detection.get("python_top3", [])
        ],
        "gtm_class": detection.get("gtm_class"),
        "gtm_class_name": display_name(detection.get("gtm_class") or "unknown"),
        "gtm_confidence": detection.get("gtm_confidence"),
        "gtm_top3": [
            [display_name(name), value] for name, value in detection.get("gtm_top3", [])
        ],
        "agreement_status": detection.get("agreement_status"),
        "confidence_difference": detection.get("confidence_difference"),
        "top_two_margin": detection.get("top_two_margin"),
        "final_class": detection.get("final_class"),
        "final_class_name": detection.get("final_class_name"),
        "confidence_level": detection.get("confidence_level"),
        "quality": detection.get("quality"),
        "quality_notes": detection.get("quality_notes"),
        "noise_level": detection.get("noise_level"),
        "severity": detection.get("severity"),
        "alert_status": detection.get("alert_status"),
        "recommended_action": detection.get("recommended_action"),
        "manual_review_required": detection.get("manual_review_required"),
        "manual_review_reason": detection.get("manual_review_reason"),
        "repeat_count": detection.get("repeat_count"),
        "overlapping_classes": [
            display_name(name) for name in detection.get("overlapping_classes", [])
        ],
        "waveform_path": detection.get("waveform_path"),
        "spectrogram_path": detection.get("spectrogram_path"),
        "model_available": detection.get("model_available", True),
        "model_message": detection.get("model_message"),
        "status": detection.get("status"),
    }


# The browser posts the Teachable Machine scores here. The Python prediction is
# never sent to the browser before this call, so it cannot influence the GTM
# result.
@app.route("/api/gtm/<int:detection_id>", methods=["POST"])
def receive_gtm(detection_id):
    user = current_user()

    if user is None:
        return jsonify({"error": "You are not signed in."}), 401

    payload = request.get_json(silent=True) or {}
    raw_scores = payload.get("scores")

    if not isinstance(raw_scores, dict) or not raw_scores:
        return jsonify({"error": "No GTM scores were received."}), 400

    try:
        result, error = analysis.apply_gtm_scores(detection_id, raw_scores)
    except Exception as error_object:
        return jsonify({"error": f"The GTM result could not be applied: {error_object}"}), 500

    if error:
        return jsonify({"error": error}), 404

    database.add_audit(user["user_id"], "gtm_result", f"detection_id={detection_id}")

    result["detection_id"] = detection_id
    return jsonify(serialise_detection(result))


@app.route("/event-history")
def event_history():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    filters = {
        "audio_id": request.args.get("audio_id", type=int),
        "filename": (request.args.get("filename") or "").strip(),
        "sound_class": request.args.get("sound_class") or "",
        "date_from": request.args.get("date_from") or "",
        "date_to": request.args.get("date_to") or "",
        "min_confidence": request.args.get("min_confidence", type=float),
        "max_confidence": request.args.get("max_confidence", type=float),
        "severity": request.args.get("severity") or "",
        "quality": request.args.get("quality") or "",
        "status": request.args.get("status") or "",
        "user_id": request.args.get("user_id", type=int),
    }

    context = page_context()
    context["user"] = user

    return render_template(
        "event_history.html",
        detections=database.search_detections(filters),
        filters=filters,
        **context,
    )


@app.route("/manual-review", methods=["GET", "POST"])
def manual_review():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    message = None
    error_message = None

    if request.method == "POST":
        if not has_role(user, ["reviewer", "admin", "operator"]):
            error_message = "Your role is not allowed to review events."
        else:
            detection_id = request.form.get("detection_id", type=int)
            decision_value = request.form.get("decision") or ""
            corrected_class = request.form.get("corrected_class") or None
            comments = (request.form.get("comments") or "").strip()
            action = (request.form.get("recommended_action") or "").strip()

            if not detection_id or database.get_detection(detection_id) is None:
                error_message = "That event does not exist."
            elif decision_value not in ("confirmed", "corrected", "false_alarm"):
                error_message = "Choose a valid decision."
            elif decision_value == "corrected" and corrected_class not in CLASSES:
                error_message = "Choose the corrected sound class."
            else:
                database.add_review(
                    detection_id,
                    user["user_id"],
                    decision_value,
                    corrected_class if decision_value == "corrected" else None,
                    comments,
                    action,
                )
                database.add_audit(
                    user["user_id"],
                    "review",
                    f"detection_id={detection_id} decision={decision_value}",
                )
                message = f"Event {detection_id} has been reviewed."

    context = page_context()
    context["user"] = user

    return render_template(
        "manual_review.html",
        queue=database.get_review_queue(),
        message=message,
        error_message=error_message,
        **context,
    )


@app.route("/alerts", methods=["GET", "POST"])
def alerts():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    message = None
    error_message = None

    if request.method == "POST":
        if not has_role(user, ["operator", "admin", "reviewer", "maintenance"]):
            error_message = "Your role is not allowed to handle alerts."
        else:
            alert_id = request.form.get("alert_id", type=int)
            action = request.form.get("action") or ""

            if action not in ("acknowledged", "dismissed", "escalated"):
                error_message = "Choose a valid action."
            elif not alert_id:
                error_message = "That alert does not exist."
            else:
                database.update_alert_status(alert_id, action, user["user_id"])
                database.add_audit(
                    user["user_id"], "alert_action", f"alert_id={alert_id} action={action}"
                )
                message = f"Alert {alert_id} was {action}."

    context = page_context()
    context["user"] = user

    return render_template(
        "alerts.html",
        active_alerts=database.get_alerts("active"),
        alert_history=database.get_alerts(),
        message=message,
        error_message=error_message,
        **context,
    )


@app.route("/reports")
def reports():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    context = page_context()
    context["user"] = user

    return render_template(
        "reports.html",
        stats=database.get_statistics(),
        model_metrics=metrics.active_model_metrics(),
        model_comparison=metrics.all_model_metrics(),
        critical_classes=[display_name(c) for c in CRITICAL_CLASSES],
        active_model=ACTIVE_PYTHON_MODEL,
        **context,
    )


@app.route("/export/detections.csv")
def export_detections():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    if not has_role(user, ["admin"]):
        return "Only an administrator can export records.", 403

    rows = database.search_detections({}, limit=20000)

    columns = [
        "detection_id",
        "audio_id",
        "filename",
        "segment_index",
        "created_at",
        "python_class",
        "python_confidence",
        "python_model_version",
        "gtm_class",
        "gtm_confidence",
        "gtm_model_version",
        "agreement_status",
        "confidence_difference",
        "top_two_margin",
        "quality",
        "noise_level",
        "final_class",
        "confidence_level",
        "severity",
        "alert_status",
        "manual_review_required",
        "reviewer_decision",
        "status",
    ]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)

    for row in rows:
        writer.writerow([row[column] if column in row.keys() else "" for column in columns])

    database.add_audit(user["user_id"], "export", f"rows={len(rows)}")

    data = io.BytesIO(output.getvalue().encode("utf-8"))

    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="sonicsentinel_detections.csv",
    )


@app.route("/admin", methods=["GET", "POST"])
def admin():
    user = current_user()

    if user is None:
        return redirect(url_for("login"))

    if not has_role(user, ["admin"]):
        context = page_context()
        context["user"] = user
        return render_template(
            "admin.html",
            settings=database.get_settings(),
            rules=rules.load_rules(),
            audit_log=[],
            anomalies=[],
            stats=database.get_statistics(),
            message=None,
            error_message="Only an administrator can open this page.",
            **context,
        ), 403

    message = None
    error_message = None

    if request.method == "POST":
        action = request.form.get("action") or "settings"

        if action == "settings":
            saved = []

            for name in [
                "min_confidence",
                "top_two_margin",
                "max_confidence_difference",
                "noise_alert_level",
                "overlap_confidence",
            ]:
                raw = request.form.get(name)

                if raw in (None, ""):
                    continue

                try:
                    value = float(raw)
                except ValueError:
                    error_message = f"'{name}' must be a number."
                    continue

                if not 0.0 <= value <= 1.0:
                    error_message = f"'{name}' must be between 0 and 1."
                    continue

                database.save_setting(name, value)
                saved.append(name)

            for name in [
                "repeated_detection_window_sec",
                "retention_days",
                "failed_login_alert_count",
            ]:
                raw = request.form.get(name)

                if raw in (None, ""):
                    continue

                try:
                    value = int(raw)
                except ValueError:
                    error_message = f"'{name}' must be a whole number."
                    continue

                if value < 1:
                    error_message = f"'{name}' must be 1 or more."
                    continue

                database.save_setting(name, value)
                saved.append(name)

            if saved and not error_message:
                message = "Settings saved: " + ", ".join(saved) + "."

            database.add_audit(user["user_id"], "settings_update", ",".join(saved))

        elif action == "retention":
            days = database.get_settings().get("retention_days", 90)
            removed = database.delete_records_older_than(days)
            database.add_audit(user["user_id"], "retention_run", f"removed={removed}")
            message = f"{removed} record(s) older than {days} days were removed."

    context = page_context()
    context["user"] = user

    return render_template(
        "admin.html",
        settings=database.get_settings(),
        rules=rules.load_rules(),
        audit_log=database.get_audit_log(40),
        anomalies=find_anomalies(),
        stats=database.get_statistics(),
        message=message,
        error_message=error_message,
        **context,
    )


# The monitoring alerts the SRS asks an administrator to be shown.
def find_anomalies():
    settings = database.get_settings()
    stats = database.get_statistics()
    notices = []

    failed_logins = database.count_recent_audit("failed_login", 3600)
    if failed_logins >= settings.get("failed_login_alert_count", 5):
        notices.append(f"{failed_logins} failed login attempts in the last hour.")

    if stats["poor_quality"] and stats["total_detections"]:
        share = stats["poor_quality"] / stats["total_detections"]
        if share > 0.3:
            notices.append(
                f"{share * 100:.0f}% of detections have poor or unusable audio quality."
            )

    if stats["total_detections"] and stats["average_confidence"] < settings.get(
        "min_confidence", 0.6
    ):
        notices.append(
            f"Average confidence ({stats['average_confidence']:.2f}) is below the minimum."
        )

    if stats["critical_alerts"] > 20:
        notices.append(f"{stats['critical_alerts']} critical alerts recorded, check for false alarms.")

    if stats["disagreements"] and stats["total_detections"]:
        share = stats["disagreements"] / stats["total_detections"]
        if share > 0.25:
            notices.append(f"The two models disagree on {share * 100:.0f}% of detections.")

    if not python_model.model_status()["available"]:
        notices.append(python_model.model_status()["message"])

    if not gtm_ready():
        notices.append("No Teachable Machine model is configured.")

    return notices


@app.route("/gtm_model/<path:filename>")
def gtm_model_file(filename):
    safe_name = secure_filename(filename)
    path = os.path.join("gtm_model", safe_name)

    if not os.path.exists(path):
        return "That GTM model file is not present.", 404

    return send_file(os.path.abspath(path))


@app.errorhandler(404)
def not_found(error):
    context = page_context()
    return render_template("error.html", code=404, message="That page does not exist.", **context), 404


@app.errorhandler(413)
def too_large(error):
    context = page_context()
    return render_template(
        "error.html",
        code=413,
        message=f"The file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        **context,
    ), 413


@app.errorhandler(500)
def server_error(error):
    context = page_context()
    return render_template(
        "error.html",
        code=500,
        message="Something went wrong while handling that request.",
        **context,
    ), 500


if __name__ == "__main__":
    app.run(debug=True)
