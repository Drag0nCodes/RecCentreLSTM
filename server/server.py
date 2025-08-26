# flask_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS module
import os
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta

# Import the LSTMModel from your local file
try:
    from model import LSTMModel
except ImportError:
    print("Error: model.py not found or LSTMModel class not defined. Please ensure 'model.py' is in the same directory.")
    exit(1)

# --- Flask App and LSTM Model Setup ---
app = Flask(__name__)
# Enable CORS for all domains, allowing your HTML page to connect
CORS(app)

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD_PATH = os.path.join(SCRIPT_DIR, 'lstm_wr_model.pth')
SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')
WR_SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, 'wr_scaler.joblib')

INPUT_SIZE = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 5

model = None
scaler = None
wr_scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """
    Loads the model and scalers once when the server starts.
    This is a better approach than using a decorator that is now deprecated.
    """
    global model, scaler, wr_scaler
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        model.to(device)
        model.eval()

        scaler = joblib.load(SCALER_LOAD_PATH)
        wr_scaler = joblib.load(WR_SCALER_LOAD_PATH)
        print("Model and scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        model = False # Set to False to indicate a loading failure

def getDateFeatures(year, month, day):
    """Helper function to get date features."""
    try:
        dt = datetime(year, month, day)
        weekOfYear = dt.isocalendar()[1]
        dayOfWeek = dt.isoweekday()
        return weekOfYear, dayOfWeek
    except ValueError:
        return None, None

@app.route('/', methods=['POST'])
def make_prediction():
    """Endpoint for making a WR value prediction."""
    if not model:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        data = request.form
        year, month, day = map(int, data.get('submissionDate').split('-'))
        startHour = int(data.get('submissionHour'))
        startDt = datetime(year, month, day, startHour)

        # Your prediction logic from the previous server.py file
        sequenceDataUnscaled = []
        wr_values_input = [float(data.get(f'integer{i}')) for i in range(1, 6)]

        for i in range(5):
            currentDt = startDt - timedelta(hours=(4 - i))
            weekOfYear, dayOfWeek = getDateFeatures(currentDt.year, currentDt.month, currentDt.day)
            if weekOfYear is None:
                return jsonify({"error": "Failed to calculate date features."}), 400
            sequenceDataUnscaled.append([
                currentDt.month, weekOfYear, currentDt.day,
                currentDt.hour, dayOfWeek, wr_values_input[i]
            ])
        
        initialSequenceUnscaled = np.array(sequenceDataUnscaled, dtype=np.float32)
        initialSequenceScaled = scaler.transform(initialSequenceUnscaled)
        
        predictedHours = []
        predictedWrs = []
        predictedWrs.append(wr_values_input[-1]) # Add last known value
        predictedHours.append(startHour)

        currentSequenceScaled = initialSequenceScaled.copy()

        with torch.no_grad():
            for predHourOffset in range(1, 24 - startHour):
                predDt = startDt + timedelta(hours=predHourOffset)
                predYear, predMonth, predDay, predHour = predDt.year, predDt.month, predDt.day, predDt.hour
                
                seqTensor = torch.FloatTensor([currentSequenceScaled]).to(device)
                predictedWrScaled = model(seqTensor).item()
                predictedWr = wr_scaler.inverse_transform(np.array([[predictedWrScaled]]))[0, 0]

                predictedHours.append(predHour)
                predictedWrs.append(predictedWr)

                predWeek, predDotw = getDateFeatures(predYear, predMonth, predDay)
                nextFeaturesActual = [predMonth, predWeek, predDay, predHour, predDotw]
                nextStepInputUnscaled = np.append(nextFeaturesActual, predictedWr)
                nextStepInputScaled = scaler.transform(nextStepInputUnscaled.reshape(1, -1)).flatten()
                currentSequenceScaled = np.vstack((currentSequenceScaled[1:], nextStepInputScaled))
        
        graph_data = [{"hour": h, "value": wr} for h, wr in zip(predictedHours, predictedWrs)]

        response_body = {
            "message": "Prediction successful!",
            "submitted_date": data.get('submissionDate'),
            "submitted_hour": data.get('submissionHour'),
            "graph_data": graph_data
        }

        return jsonify(response_body)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

if __name__ == '__main__':
    # Load the model and scalers here, before the server starts listening
    load_model()
    # Run the Flask app on all interfaces
    app.run(host='0.0.0.0', port=8080)
