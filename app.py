import os
import json
import base64
import numpy as np
import logging
from flask import Flask, jsonify
import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
from scipy.signal import butter, filtfilt

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Decode base64 Firebase key from environment variable
try:
    firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
    if not firebase_key_b64:
        raise EnvironmentError("FIREBASE_KEY_B64 not set in environment variables.")
    
    firebase_key_json = base64.b64decode(firebase_key_b64).decode("utf-8")
    with open("firebase_key.json", "w") as f:
        f.write(firebase_key_json)
    logger.info("✅ Firebase key decoded successfully")
except Exception as e:
    logger.error(f"❌ Firebase key setup failed: {e}")
    raise

# Firebase initialization with error handling
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(
        cred, {"databaseURL": "https://goldencare-68364-default-rtdb.firebaseio.com/"}
    )
    logger.info("✅ Firebase initialized successfully")
except Exception as e:
    logger.error(f"❌ Firebase initialization failed: {e}")
    raise

# Load trained CNN model with error handling
try:
    model = tf.keras.models.load_model("ecg_model.h5", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
}

app = Flask(__name__)

# Signal processing filter settings with error handling
try:
    FS = 125
    LOW = 0.5
    HIGH = 40.0
    b, a = butter(3, [LOW / (FS / 2), HIGH / (FS / 2)], btype='band')
    logger.info("✅ Filter coefficients computed")
except Exception as e:
    logger.error(f"❌ Filter initialization failed: {e}")
    b, a = None, None

def fetch_last_two_packets():
    """Fetch last two ECG readings with detailed error handling."""
    try:
        ref = db.reference("/patients/patient_123/readings")
        nodes = ref.order_by_key().limit_to_last(2).get()
        
        logger.info(f"📥 Firebase query returned: {type(nodes)}")
        
        if not nodes:
            raise FileNotFoundError("No readings found for this patient.")
        
        if len(nodes) < 2:
            raise ValueError(f"Need at least 2 readings for prediction, found {len(nodes)}.")
        
        ts = sorted(nodes.keys(), reverse=True)
        latest = nodes[ts[0]]
        prev = nodes[ts[1]]
        
        logger.info(f"📊 Processing readings: {ts[0]} (latest), {ts[1]} (previous)")
        
        # Validate data structure
        for name, packet in [("latest", latest), ("previous", prev)]:
            if not isinstance(packet, dict):
                raise ValueError(f"{name} packet is not a valid dictionary.")
            
            if "ecg_sequence" not in packet:
                raise KeyError(f"Missing 'ecg_sequence' in {name} packet.")
            
            if not packet["ecg_sequence"]:
                raise ValueError(f"'ecg_sequence' is empty in {name} packet.")
            
            # Log sequence info
            seq = packet["ecg_sequence"]
            logger.info(f"{name} ECG sequence length: {len(seq)}")
        
        return latest, prev
        
    except Exception as e:
        logger.error(f"❌ Firebase fetch error: {e}")
        raise

def make_187beat(latest, prev):
    """Process ECG data with robust error handling."""
    try:
        logger.info("🔄 Processing ECG sequences...")
        
        # Parse sequences with validation
        try:
            ecg_latest = np.fromstring(latest["ecg_sequence"], sep=",", dtype=np.float32)
            ecg_prev = np.fromstring(prev["ecg_sequence"], sep=",", dtype=np.float32)
        except ValueError as e:
            # Try parsing as integers if float fails
            logger.warning("⚠️ Float parsing failed, trying integer parsing")
            ecg_latest = np.fromstring(latest["ecg_sequence"], sep=",", dtype=np.int16).astype(np.float32)
            ecg_prev = np.fromstring(prev["ecg_sequence"], sep=",", dtype=np.int16).astype(np.float32)
        
        logger.info(f"Parsed lengths - Latest: {len(ecg_latest)}, Previous: {len(ecg_prev)}")
        
        if len(ecg_latest) == 0:
            raise ValueError("Latest ECG sequence is empty after parsing.")
        
        if len(ecg_prev) == 0:
            raise ValueError("Previous ECG sequence is empty after parsing.")
        
        # Pad sequences if too short (ethical consideration handled)
        if len(ecg_latest) < 125:
            logger.warning(f"⚠️ Latest sequence too short ({len(ecg_latest)}), padding to 125")
            ecg_latest = np.pad(ecg_latest, (0, 125 - len(ecg_latest)), mode='edge')
        
        if len(ecg_prev) < 125:
            logger.warning(f"⚠️ Previous sequence too short ({len(ecg_prev)}), padding to 125")
            ecg_prev = np.pad(ecg_prev, (0, 125 - len(ecg_prev)), mode='edge')
        
        # Take only first 125 samples if longer
        ecg_latest = ecg_latest[:125]
        ecg_prev = ecg_prev[:125]
        
        # Combine and take last 187 samples
        raw = np.concatenate([ecg_prev, ecg_latest])[-187:].astype("float32")
        
        if len(raw) < 187:
            raise ValueError(f"Not enough samples for 187-beat sequence: got {len(raw)}")
        
        # Apply filtering if available
        if b is not None and a is not None:
            raw = filtfilt(b, a, raw)
        else:
            logger.warning("⚠️ Skipping filtering due to filter initialization failure")
        
        # Normalize to [-1,1]
        signal_range = raw.max() - raw.min()
        if signal_range == 0:
            logger.warning("⚠️ Zero signal range detected, using default normalization")
            beat = np.zeros_like(raw)
        else:
            beat = (raw - raw.min()) / (signal_range + 1e-7)
            beat = 2.0 * beat - 1.0
        
        logger.info(f"✅ Signal processed successfully, shape: {beat.shape}")
        return beat.reshape(1, 187, 1)
        
    except Exception as e:
        logger.error(f"❌ Signal processing failed: {e}")
        raise RuntimeError(f"Signal processing failed: {str(e)}")

@app.route("/predict", methods=["GET"])
def predict():
    """Main prediction endpoint with comprehensive error handling."""
    logger.info("🚀 Prediction request received")
    
    # Check if model is loaded
    if model is None:
        logger.error("❌ Model not available")
        return jsonify({
            "error_code": 500, 
            "error_message": "Model not loaded. Check server logs."
        }), 500
    
    try:
        # Step 1: Fetch data
        logger.info("Step 1: Fetching data...")
        latest, prev = fetch_last_two_packets()
        
        # Step 2: Process signal
        logger.info("Step 2: Processing signal...")
        x = make_187beat(latest, prev)
        
        # Step 3: Run prediction
        logger.info("Step 3: Running prediction...")
        probs = model.predict(x, batch_size=1, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        
        logger.info(f"✅ Prediction: {CLASS_NAMES[idx]} ({confidence:.4f})")
        
        return jsonify({
            "prediction": CLASS_NAMES[idx],
            "confidence": round(confidence, 4)
        }), 200
        
    except FileNotFoundError as e:
        logger.error(f"❌ Data not found: {e}")
        return jsonify({"error_code": 404, "error_message": str(e)}), 404
    except (KeyError, ValueError) as e:
        logger.error(f"❌ Data validation error: {e}")
        return jsonify({"error_code": 400, "error_message": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return jsonify({
            "error_code": 500, 
            "error_message": f"Server error: {str(e)}"
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "filter_ready": b is not None and a is not None
    }), 200

if __name__ == "__main__":
    logger.info("🚀 Starting ECG Classification API...")
    app.run(host="0.0.0.0", port=8080, debug=False)
