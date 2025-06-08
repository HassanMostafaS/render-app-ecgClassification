old
import os
import json
import base64
import numpy as np
from flask import Flask, jsonify
import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
from scipy.signal import butter, filtfilt

# Decode base64 Firebase key from environment variable
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
if not firebase_key_b64:
    raise EnvironmentError("FIREBASE_KEY_B64 not set in environment variables.")

firebase_key_json = base64.b64decode(firebase_key_b64).decode("utf-8")
with open("firebase_key.json", "w") as f:
    f.write(firebase_key_json)

# Firebase initialization
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(
    cred, {"databaseURL": "https://goldencare-68364-default-rtdb.firebaseio.com/"}
)

# Load trained CNN model
model = tf.keras.models.load_model("ecg_model.h5")

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
}

app = Flask(_name_)

# Signal processing filter settings
FS = 125
LOW = 0.5
HIGH = 40.0
b, a = butter(3, [LOW / (FS / 2), HIGH / (FS / 2)], btype='band')

def fetch_last_two_packets():
    ref = db.reference("/patients/patient_123/readings")
    nodes = ref.order_by_key().limit_to_last(2).get()

    if not nodes:
        raise FileNotFoundError("No readings found for this patient.")

    if len(nodes) < 2:
        raise ValueError("Need at least 2 readings for prediction.")

    ts = sorted(nodes.keys(), reverse=True)
    latest = nodes[ts[0]]
    prev = nodes[ts[1]]

    for name, packet in [("latest", latest), ("previous", prev)]:
        if "ecg_sequence" not in packet:
            raise KeyError(f"Missing 'ecg_sequence' in {name} packet.")
        if not packet["ecg_sequence"]:
            raise ValueError(f"'ecg_sequence' is empty in {name} packet.")

    return latest, prev

def make_187beat(latest, prev):
    try:
        ecg_latest = np.fromstring(latest["ecg_sequence"], sep=",", dtype=np.float32)
        ecg_prev = np.fromstring(prev["ecg_sequence"], sep=",", dtype=np.float32)

        raw = np.concatenate([ecg_prev, ecg_latest])[-187:].astype("float32")
        if len(raw) < 187:
            raise ValueError("Not enough samples for 187-beat sequence.")

        raw = filtfilt(b, a, raw)

        beat = (raw - raw.min()) / (raw.max() - raw.min() + 1e-7)
        beat = 2.0 * beat - 1.0
        return beat.reshape(1, 187, 1)

    except Exception as e:
        raise RuntimeError(f"Signal processing failed: {str(e)}")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        latest, prev = fetch_last_two_packets()
        x = make_187beat(latest, prev)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return jsonify({
            "prediction": CLASS_NAMES[idx],
            "confidence": round(float(probs[idx]), 4)
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error_code": 404, "error_message": str(e)}), 404
    except (KeyError, ValueError) as e:
        return jsonify({"error_code": 400, "error_message": str(e)}), 400
    except Exception as e:
        return jsonify({"error_code": 500, "error_message": f"Server error: {str(e)}"}), 500

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=8080)
