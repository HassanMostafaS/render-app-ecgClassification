import os
import json
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

# Direct Firebase credential setup using JSON file
def setup_firebase_credentials():
    """Setup Firebase credentials by reading serviceAccountKey.json file."""
    try:
        # Try multiple possible file locations
        possible_paths = [
            "serviceAccountKey.json",
            "./serviceAccountKey.json",
            "/app/serviceAccountKey.json",  # Common deployment path
            os.path.join(os.getcwd(), "serviceAccountKey.json")
        ]
        
        service_account_file = None
        for path in possible_paths:
            if os.path.exists(path):
                service_account_file = path
                logger.info(f"✅ Found service account file at: {path}")
                break
        
        if not service_account_file:
            raise FileNotFoundError(f"serviceAccountKey.json not found in any of these locations: {possible_paths}")
        
        # Validate the JSON file
        try:
            with open(service_account_file, 'r') as f:
                firebase_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in serviceAccountKey.json: {e}")
        except Exception as e:
            raise ValueError(f"Error reading serviceAccountKey.json: {e}")
        
        # Validate required fields
        required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email"]
        missing_fields = [field for field in required_fields if field not in firebase_data]
        if missing_fields:
            raise ValueError(f"Missing required fields in serviceAccountKey.json: {missing_fields}")
        
        logger.info(f"✅ Service account validated for project: {firebase_data.get('project_id')}")
        return service_account_file
        
    except Exception as e:
        logger.error(f"❌ Firebase credential setup failed: {e}")
        raise

# Initialize Firebase
try:
    firebase_cred_file = setup_firebase_credentials()
    
    # Initialize Firebase with the JSON file
    cred = credentials.Certificate(firebase_cred_file)
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://goldencare-68364-default-rtdb.firebaseio.com/"
    })
    
    # Test Firebase connection
    logger.info("🔄 Testing Firebase connection...")
    test_ref = db.reference("/")
    test_ref.get()  # This will fail immediately if credentials are wrong
    
    logger.info("✅ Firebase initialized and tested successfully")
    
except Exception as e:
    logger.error(f"❌ Firebase initialization failed: {e}")
    raise

# Load trained CNN model
try:
    # Try multiple possible model file locations
    model_paths = [
        "ecg_model.h5",
        "./ecg_model.h5",
        "/app/ecg_model.h5",
        "best_ecg_model.h5",
        "checkpoint_model2.h5"
    ]
    
    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            logger.info(f"✅ Found model file at: {path}")
            break
    
    if not model_file:
        raise FileNotFoundError(f"Model file not found in any of these locations: {model_paths}")
    
    model = tf.keras.models.load_model(model_file, compile=False)
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

# Signal processing filter settings
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
    """Fetch last two ECG readings."""
    try:
        logger.info("📥 Fetching data from Firebase...")
        
        ref = db.reference("/patients/patient_123/readings")
        nodes = ref.order_by_key().limit_to_last(2).get()
        
        if not nodes:
            raise FileNotFoundError("No readings found for this patient.")
        
        if len(nodes) < 2:
            raise ValueError(f"Need at least 2 readings for prediction, found {len(nodes)}.")
        
        ts = sorted(nodes.keys(), reverse=True)
        latest = nodes[ts[0]]
        prev = nodes[ts[1]]
        
        # Validate data structure
        for name, packet in [("latest", latest), ("previous", prev)]:
            if not isinstance(packet, dict):
                raise ValueError(f"{name} packet is not a valid dictionary.")
            
            if "ecg_sequence" not in packet:
                raise KeyError(f"Missing 'ecg_sequence' in {name} packet.")
            
            if not packet["ecg_sequence"]:
                raise ValueError(f"'ecg_sequence' is empty in {name} packet.")
        
        logger.info(f"✅ Data fetched successfully: {ts[0]} (latest), {ts[1]} (previous)")
        return latest, prev
        
    except Exception as e:
        logger.error(f"❌ Firebase fetch error: {e}")
        raise

def make_187beat(latest, prev):
    """Process ECG data."""
    try:
        logger.info("🔄 Processing ECG sequences...")
        
        # Parse sequences - try float first, then int
        try:
            ecg_latest = np.fromstring(latest["ecg_sequence"], sep=",", dtype=np.float32)
            ecg_prev = np.fromstring(prev["ecg_sequence"], sep=",", dtype=np.float32)
        except ValueError:
            logger.info("⚠️ Float parsing failed, trying integer parsing")
            ecg_latest = np.fromstring(latest["ecg_sequence"], sep=",", dtype=np.int16).astype(np.float32)
            ecg_prev = np.fromstring(prev["ecg_sequence"], sep=",", dtype=np.int16).astype(np.float32)
        
        if len(ecg_latest) == 0 or len(ecg_prev) == 0:
            raise ValueError("One or both ECG sequences are empty after parsing.")
        
        # Pad sequences if too short
        if len(ecg_latest) < 125:
            logger.warning(f"⚠️ Latest sequence too short ({len(ecg_latest)}), padding to 125")
            ecg_latest = np.pad(ecg_latest, (0, 125 - len(ecg_latest)), mode='edge')
        
        if len(ecg_prev) < 125:
            logger.warning(f"⚠️ Previous sequence too short ({len(ecg_prev)}), padding to 125")
            ecg_prev = np.pad(ecg_prev, (0, 125 - len(ecg_prev)), mode='edge')
        
        # Take exactly 125 samples
        ecg_latest = ecg_latest[:125]
        ecg_prev = ecg_prev[:125]
        
        # Combine and take last 187
        raw = np.concatenate([ecg_prev, ecg_latest])[-187:].astype("float32")
        
        # Apply filtering
        if b is not None and a is not None:
            raw = filtfilt(b, a, raw)
        else:
            logger.warning("⚠️ Skipping filtering due to filter initialization failure")
        
        # Normalize
        signal_range = raw.max() - raw.min()
        if signal_range == 0:
            logger.warning("⚠️ Zero signal range detected")
            beat = np.zeros_like(raw)
        else:
            beat = (raw - raw.min()) / (signal_range + 1e-7)
            beat = 2.0 * beat - 1.0
        
        logger.info("✅ Signal processed successfully")
        return beat.reshape(1, 187, 1)
        
    except Exception as e:
        logger.error(f"❌ Signal processing failed: {e}")
        raise RuntimeError(f"Signal processing failed: {str(e)}")

@app.route("/predict", methods=["GET"])
def predict():
    """Main prediction endpoint."""
    if model is None:
        return jsonify({
            "error_code": 500, 
            "error_message": "Model not loaded"
        }), 500
    
    try:
        latest, prev = fetch_last_two_packets()
        x = make_187beat(latest, prev)
        probs = model.predict(x, batch_size=1, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        
        logger.info(f"✅ Prediction: {CLASS_NAMES[idx]} ({confidence:.4f})")
        
        return jsonify({
            "prediction": CLASS_NAMES[idx],
            "confidence": round(confidence, 4)
        }), 200
        
    except FileNotFoundError as e:
        return jsonify({"error_code": 404, "error_message": str(e)}), 404
    except (KeyError, ValueError) as e:
        return jsonify({"error_code": 400, "error_message": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({
            "error_code": 500, 
            "error_message": f"Server error: {str(e)}"
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        # Test Firebase connection
        ref = db.reference("/patients/patient_123/readings")
        test_data = ref.order_by_key().limit_to_last(1).get()
        firebase_ok = test_data is not None
    except:
        firebase_ok = False
    
    return jsonify({
        "status": "healthy" if (model is not None and firebase_ok) else "degraded",
        "model_loaded": model is not None,
        "firebase_connected": firebase_ok,
        "filter_ready": b is not None and a is not None
    }), 200

@app.route("/", methods=["GET"])
def info():
    """API info endpoint."""
    return jsonify({
        "service": "ECG Classification API",
        "status": "running",
        "endpoints": {
            "/predict": "GET - ECG classification",
            "/health": "GET - Health check",
            "/": "GET - API info"
        }
    })

if __name__ == "__main__":
    logger.info("🚀 Starting ECG Classification API...")
    app.run(host="0.0.0.0", port=8080, debug=False)
