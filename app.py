import os
import numpy as np
from flask import Flask, jsonify

import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
from scipy.signal import butter, filtfilt

# Firebase setup
cred = credentials.Certificate(
    r"C:\Graduation_Project\lstm\serviceAccountKey.json"
)
firebase_admin.initialize_app(
    cred,
    {"databaseURL": "https://goldencare-68364-default-rtdb.firebaseio.com/"}
)

# Load 5-class CNN model
model = tf.keras.models.load_model(
    r"ecg_model.h5"
)

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
}

app = Flask(__name__)

# Signal filtering settings
FS = 125  # Hz
LOW = 0.5  # Hz
HIGH = 40.0  # Hz
b, a = butter(3, [LOW / (FS / 2), HIGH / (FS / 2)], btype='band')

def fetch_last_two_packets():
    ref = db.reference(f"/patients/patient_123/readings")
    nodes = ref.order_by_key().limit_to_last(2).get()

    if not nodes:
        raise FileNotFoundError("No readings found for this patient in the database.")

    if len(nodes) < 2:
        raise ValueError("At least 2 readings are required to make a prediction.")

    ts = sorted(nodes.keys(), reverse=True)
    latest = nodes[ts[0]]
    prev = nodes[ts[1]]

    for label, packet in [("latest", latest), ("previous", prev)]:
        if "ecg_sequence" not in packet:
            raise KeyError(f"Missing 'ecg_sequence' in {label} packet")
        if not packet["ecg_sequence"]:
            raise ValueError(f"'ecg_sequence' is empty in {label} packet")

    return latest, prev

def make_187beat(latest, prev):
    """Build 187-sample beat, filter, scale to [-1,1]."""
    try:
        ecg_latest = np.fromstring(latest["ecg_sequence"], sep=",", dtype=np.float32)
        ecg_prev = np.fromstring(prev["ecg_sequence"], sep=",", dtype=np.float32)

        # print(f"Latest ECG samples: {len(ecg_latest)}")
        # print(f"Prev ECG samples: {len(ecg_prev)}")

        raw = np.concatenate([ecg_prev, ecg_latest])[-187:].astype("float32")

        if len(raw) < 187:
            raise ValueError(f"Not enough data: need 187 samples, got {len(raw)}")

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
    except KeyError as e:
        return jsonify({"error_code": 400, "error_message": str(e)}), 400
    except ValueError as e:
        return jsonify({"error_code": 400, "error_message": str(e)}), 400
    except Exception as e:
        return jsonify({"error_code": 500, "error_message": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run()
