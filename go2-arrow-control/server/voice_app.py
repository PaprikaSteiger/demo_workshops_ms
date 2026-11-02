"""
GO2 Arrow Control with the voice extension.
Serves everything app.py serves, plus the voice pages under /voice.
Run from the server/ folder: python voice_app.py
"""

from app import app, control_logger
from voice import voice

app.register_blueprint(voice)

if __name__ == "__main__":
    control_logger.info("Voice pages: https://localhost:5000/voice/sim and https://localhost:5000/voice/control")
    # Same as app.py: ad-hoc SSL so browsers allow camera and microphone access over the network
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context="adhoc")
