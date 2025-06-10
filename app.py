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

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="ecg_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown"
}

app = Flask(__name__)

# Signal processing filter settings
FS = 125
LOW = 0.5
HIGH = 40.0
b, a = butter(3, [LOW / (FS / 2), HIGH / (FS / 2)], btype='band')

def fetch_last_two_packets():
    ref = db.reference("/patients/patient_123/readings")
    nodes = ref.order_by_key().limit_to_last(4).get()

    if not nodes:
        raise FileNotFoundError("No readings found for this patient.")

    if len(nodes) < 2:
        raise ValueError("Need at least 2 readings for prediction.")

    ts = sorted(nodes.keys(), reverse=True)
    prev3 = nodes[ts[2]]
    prev4 = nodes[ts[3]]

    for name, packet in [("prev3", prev3), ("prev4", prev4)]:
        if "ecg_sequence" not in packet:
            raise KeyError(f"Missing 'ecg_sequence' in {name} packet.")
        if not packet["ecg_sequence"]:
            raise ValueError(f"'ecg_sequence' is empty in {name} packet.")

    return prev3, prev4

def make_187beat(prev3, prev4):
    try:
        ecg_prev3 = np.fromstring(prev3["ecg_sequence"], sep=",", dtype=np.float32)
        ecg_prev4 = np.fromstring(prev4["ecg_sequence"], sep=",", dtype=np.float32)

        raw = np.concatenate([ecg_prev4, ecg_prev3])[-187:].astype("float32")
        if len(raw) < 187:
            raise ValueError("Not enough samples for 187-beat sequence.")

        filtered = filtfilt(b, a, raw)

        beat = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-7)
        beat = 2.0 * beat - 1.0
        return beat.reshape(1, 187, 1).astype(np.float32)

    except Exception as e:
        raise RuntimeError(f"Signal processing failed: {str(e)}")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        prev3, prev4 = fetch_last_two_packets()
        x = make_187beat(prev3, prev4)

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(np.argmax(probs))
        return jsonify({
            "prediction": CLASS_NAMES[idx]
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error_code": 404, "error_message": str(e)}), 404
    except (KeyError, ValueError) as e:
        return jsonify({"error_code": 400, "error_message": str(e)}), 400
    except Exception as e:
        return jsonify({"error_code": 500, "error_message": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
