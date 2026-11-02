"""
GO2 Voice Control - extension for Teachable Machine audio models
Adds the voice pages (/voice/sim, /voice/control) and their routes to the app in app.py
without changing it. Started via voice_app.py.
"""

import os
import json
import math
import time
import zipfile
import threading
from io import BytesIO
from collections import deque
from datetime import datetime

from flask import Blueprint, render_template, request, jsonify, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename

import app as core

voice = Blueprint("voice", __name__, url_prefix="/voice")

# Separate from uploads/models, so the image pages never list audio models
VOICE_UPLOAD_FOLDER = "uploads/voice_models"
# Own leaderboard: voice reacts about a second slower than arrows
VOICE_MAZE_ID = "voice-maze-1"

os.makedirs(VOICE_UPLOAD_FOLDER, exist_ok=True)

# --- Robot steps ---
# Voice can't give a continuous signal like a held-up arrow, so on the robot one confirmed word
# = one fixed step. Forward/Backwards in metres, Left/Right in degrees. The duration follows from
# the speeds the robot uses (robot_controller.py: forward = max_speed, backwards = reverse_ratio x,
# turning = turn_ratio x rad/s), so Max Speed changes how fast a step is, not how far.
ROBOT_STEPS = {"Forward": 0.3, "Backwards": 0.3, "Left": 45, "Right": 45}
# The real robot accelerates and brakes: raise a factor if it falls short (1.0 = as calculated)
STEP_CALIBRATION = {"Forward": 1.0, "Backwards": 1.0, "Left": 1.0, "Right": 1.0}
STEP_RESEND_INTERVAL = 0.1  # seconds between repeated move commands during a step

# Same decision rule as the voice simulator (VOICE_RULE in voice_sim.js)
VOICE_RULE = {"min_confidence": 0.75, "window": 3, "votes_required": 2}
# One word stays in the results for ~1-1.5 s: the same word is ignored this long after its step started
REPEAT_GUARD = 1.5

step_lock = threading.Lock()
step_state = {"command": None, "last": None, "started": 0.0}  # command is set while a step runs
vote_buffer = deque(maxlen=VOICE_RULE["window"])


# --- Model storage ---
# Each model is a Teachable Machine audio export (speech-commands), extracted to
# VOICE_UPLOAD_FOLDER/<name>_<ts>/ (model.json, metadata.json, weights.bin). It runs in the browser.
# The class names are in metadata.json "wordLabels" (image models use "labels").


def save_voice_model(file, base_name):
    """Extract an uploaded Teachable Machine audio .zip. Returns the stored model name."""
    try:
        zf = zipfile.ZipFile(BytesIO(file.read()))
    except zipfile.BadZipFile:
        raise ValueError("The file is not a valid .zip")

    # Index members by file name, ignoring folders and macOS metadata
    members = {}
    for name in zf.namelist():
        base = os.path.basename(name)
        if base and not name.startswith("__MACOSX/") and not base.startswith("._"):
            members.setdefault(base, name)

    if "model.json" not in members or "metadata.json" not in members:
        raise ValueError(
            "This zip doesn't look like a Teachable Machine Tensorflow.js export "
            "(expected model.json + metadata.json + weights.bin)"
        )

    model_json = json.loads(zf.read(members["model.json"]))
    metadata = json.loads(zf.read(members["metadata.json"]))
    if not metadata.get("wordLabels"):
        raise ValueError(
            "metadata.json has no wordLabels. Is this an image model? Upload it on the arrow pages instead."
        )
    weight_files = [
        os.path.basename(p)
        for group in model_json.get("weightsManifest", [])
        for p in group.get("paths", [])
    ]
    missing = [w for w in weight_files if w not in members]
    if missing:
        raise ValueError(f"The zip is missing {', '.join(missing)}")

    target = os.path.join(VOICE_UPLOAD_FOLDER, base_name)
    os.makedirs(target)
    for fname in ["model.json", "metadata.json"] + weight_files:
        with open(os.path.join(target, fname), "wb") as out:
            out.write(zf.read(members[fname]))
    return base_name


def get_voice_model_info(filename):
    """Describe a stored voice model, in the same format as get_model_info() in app.py."""
    filename = os.path.basename(filename)
    path = os.path.join(VOICE_UPLOAD_FOLDER, filename)
    if not filename or not os.path.isfile(os.path.join(path, "metadata.json")):
        return None

    with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
        labels = json.load(f).get("wordLabels", [])

    # Stored as safe_name_YYYYMMDD_HHMMSS: strip the timestamp for display
    parts = filename.rsplit("_", 2)
    name = parts[0].replace("_", " ") if len(parts) == 3 else filename

    commands, recognised, warnings, method = core.map_labels(labels)
    return {
        "filename": filename,
        "name": name,
        "labels": labels,
        "commands": commands,
        "recognised": recognised,
        "warnings": warnings,
        "mapping_method": method,
        "model_type": "audio",
        "size": sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)),
        "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
        "model_url": url_for("voice.model_file", model=filename, file="model.json"),
        "metadata_url": url_for("voice.model_file", model=filename, file="metadata.json"),
    }


# --- Pages ---


@voice.route("/sim")
def simulator():
    """Voice maze simulator: test an audio model without the robot"""
    return render_template("voice_sim.html")


@voice.route("/control")
def control_page():
    """Robot control with an audio model"""
    return render_template("voice_control.html")


# --- Models ---


@voice.route("/upload_model", methods=["POST"])
def upload_model():
    """Upload a Teachable Machine audio Tensorflow.js export (.zip)"""
    file = request.files.get("model")
    model_name = request.form.get("model_name", "").strip()

    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400
    if not core.allowed_file(file.filename):
        return jsonify({"error": "Please upload the Teachable Machine Tensorflow.js .zip"}), 400

    try:
        # Name with timestamp: safe_name_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = secure_filename(model_name) or "model"
        filename = save_voice_model(file, f"{safe_name}_{timestamp}")
        info = get_voice_model_info(filename)

        core.control_logger.info(
            "Uploaded model: %s (%s, voice, labels: %s)", model_name, filename, info["labels"]
        )
        return jsonify(
            {"success": True, "filename": filename, "model_name": model_name, "model": info}
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@voice.route("/models", methods=["GET"])
def list_models():
    """List all uploaded voice models"""
    models = []
    for filename in os.listdir(VOICE_UPLOAD_FOLDER):
        try:
            info = get_voice_model_info(filename)
        except Exception as e:
            core.control_logger.warning("Skipping unreadable voice model %s: %s", filename, e)
            continue
        if info:
            models.append(info)

    models.sort(key=lambda m: m["modified"], reverse=True)
    return jsonify({"models": models, "current": core.current_model_name})


@voice.route("/model_files/<model>/<path:file>")
def model_file(model, file):
    """Serve the speech-commands model files to the browser"""
    folder = os.path.abspath(os.path.join(VOICE_UPLOAD_FOLDER, os.path.basename(model)))
    if not os.path.isdir(folder):
        abort(404)
    return send_from_directory(folder, file)


@voice.route("/load_model", methods=["POST"])
def load_model():
    """Make a voice model the robot's model, like /load_model does for image models.
    The pilot's browser then sends its class probabilities to /submit_prediction as usual."""
    if not core.is_current_pilot():
        return jsonify({"error": "Not the current pilot"}), 403

    info = get_voice_model_info(request.form.get("filename") or "")
    if info is None:
        return jsonify({"error": "Model file not found"}), 404

    core.current_model_name = info["filename"]
    core.current_model_info = info
    core.control_logger.info("Loaded model: %s (voice)", info["filename"])

    return jsonify({"success": True, "message": f"Loaded model: {info['filename']}", "model": info})


# --- Robot control: one confirmed word = one step ---


def step_duration(command):
    """Seconds one step takes at the current Max Speed."""
    speed = float(core.settings["max_speed"])
    rc = core.robot_controller
    if command == "Forward":
        seconds = ROBOT_STEPS[command] / speed
    elif command == "Backwards":
        seconds = ROBOT_STEPS[command] / (speed * rc.reverse_ratio)
    else:
        seconds = math.radians(ROBOT_STEPS[command]) / (speed * rc.turn_ratio)
    return seconds * STEP_CALIBRATION[command]


def run_step(command, duration):
    """Repeat the move until the step is done, then stop. Aborts as soon as control is stopped
    (Stop, Emergency Stop, Relinquish and the teacher page all switch inference off)."""
    rc = core.robot_controller
    end = time.time() + duration
    try:
        while core.settings["inference_enabled"]:
            remaining = end - time.time()
            if remaining <= 0:
                break
            rc.execute_command(command, core.settings["max_speed"])
            time.sleep(min(STEP_RESEND_INTERVAL, remaining))
        rc.stop()
    finally:
        with step_lock:
            step_state["command"] = None


@voice.route("/submit_prediction", methods=["POST"])
def submit_prediction():
    """Like /submit_prediction in app.py, but a confirmed word starts one fixed robot step.
    Expects: JSON {"probabilities": [float per class]} from the pilot's browser."""
    if not core.is_current_pilot():
        return jsonify({"error": "Not the current pilot"}), 403
    if not (core.model_ready() and core.settings["inference_enabled"]):
        return (
            jsonify(
                {
                    "error": "Inference not enabled or no model loaded",
                    "model_loaded": core.model_ready(),
                    "inference_enabled": core.settings["inference_enabled"],
                }
            ),
            400,
        )

    data = request.get_json(silent=True) or {}
    probabilities = data.get("probabilities")
    labels = core.current_model_info["labels"]
    commands = core.current_model_info["commands"]
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != len(labels)
        or not all(isinstance(p, (int, float)) for p in probabilities)
    ):
        return jsonify({"error": "probabilities must list one number per class", "model_changed": True}), 400
    core.pilot_last_active = time.time()

    top = max(range(len(probabilities)), key=lambda i: probabilities[i])
    confidence = float(probabilities[top])
    # Only confident results vote for a move; everything else votes Idle
    vote = commands[top] if confidence >= VOICE_RULE["min_confidence"] else "Idle"

    now = time.time()
    command = "Idle"
    votes = 0
    with step_lock:
        # While a step runs, words are ignored: steps are never cut short on the real robot
        if step_state["command"] is None:
            vote_buffer.append(vote)
            votes = vote_buffer.count(vote)
            repeat = vote == step_state["last"] and now - step_state["started"] < REPEAT_GUARD
            if vote != "Idle" and votes >= VOICE_RULE["votes_required"] and not repeat:
                command = vote
                vote_buffer.clear()
                step_state.update(command=command, last=command, started=now)

    if command != "Idle":
        duration = step_duration(command)
        unit = "m" if command in ("Forward", "Backwards") else "°"
        core.control_logger.info(
            "Voice step: %s %s%s in %.2fs", command, ROBOT_STEPS[command], unit, duration
        )
        threading.Thread(target=run_step, args=(command, duration), daemon=True).start()

    probabilities = [float(p) for p in probabilities]
    # For the Pilot View of other groups, like process_probabilities() in app.py
    core.LAST_PREDICTION_DATA = {
        "prediction": labels[top],
        "confidence": confidence,
        "command_to_execute": command,
        "labels": labels,
        "probabilities": probabilities,
    }

    return jsonify(
        {
            "prediction": labels[top],
            "confidence": confidence,
            "labels": labels,
            "probabilities": probabilities,
            "votes": votes,
            "votes_required": VOICE_RULE["votes_required"],
            "command_to_execute": command,
        }
    )


# --- Simulator leaderboard ---
# Same storage as in app.py, under VOICE_MAZE_ID. Read it with GET /api/leaderboard?maze_id=voice-maze-1.


@voice.route("/leaderboard", methods=["POST"])
def submit_leaderboard_time():
    """Record a finished voice simulator run. Only a better time changes the leaderboard."""
    data = request.get_json(silent=True) or {}

    info = get_voice_model_info(str(data.get("filename") or ""))
    if info is None:
        return jsonify({"error": "Model not found"}), 404

    try:
        time_ms = int(data.get("time_ms"))
    except (TypeError, ValueError):
        return jsonify({"error": "time_ms must be a number"}), 400
    lo, hi = core.SIM_TIME_LIMITS_MS
    if time_ms < lo or time_ms > hi:
        return jsonify({"error": f"time_ms must be between {lo} and {hi}"}), 400

    name = info["name"]
    today = datetime.now().strftime("%Y-%m-%d")

    with core.LEADERBOARD_LOCK:
        board = core.read_leaderboard()
        days = board.setdefault(VOICE_MAZE_ID, {}).setdefault(name, {})
        previous_all = min(days.values()) if days else None
        previous_today = days.get(today)

        new_best_today = previous_today is None or time_ms < previous_today
        if new_best_today:
            days[today] = time_ms
            core.write_leaderboard(board)

    core.control_logger.info("Voice simulator run: %s finished in %.1fs", name, time_ms / 1000)

    return jsonify(
        {
            "name": name,
            "time_ms": time_ms,
            "best_today_ms": days[today],
            "best_all_ms": min(days.values()),
            "new_best_today": new_best_today,
            "new_best_all": previous_all is None or time_ms < previous_all,
        }
    )
