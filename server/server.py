import socket
import threading
import json
import urllib.parse
import os
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import queue

# --- Server Config ---
HOST = '127.0.0.1'
PORT = 8080

# --- LSTM Model Config (ensure these match your training) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD_PATH = os.path.join(SCRIPT_DIR, '../LSTM/lstm_wr_model.pth')
SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, '../LSTM/scaler.joblib')
WR_SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, '../LSTM/wr_scaler.joblib')

INPUT_SIZE = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 5

# --- Global variables for the prediction thread ---
prediction_queue = queue.Queue() # Queue for incoming prediction requests
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
scaler = None
wr_scaler = None
model_ready = threading.Event() # Event to signal when the model is loaded

# Assuming model.py exists in the same directory and contains LSTMModel
try:
    from model import LSTMModel
except ImportError:
    print("Error: model.py not found or LSTMModel class not defined. Please ensure 'model.py' is in the same directory.")
    # Exit or handle appropriately if model definition is critical
    exit(1)


# --- Helper functions from predict.py ---
def getDateFeatures(year, month, day):
    try:
        dt = datetime(year, month, day)
        weekOfYear = dt.isocalendar()[1]
        dayOfWeek = dt.isoweekday() # monday is 1, sunday is 7
        return weekOfYear, dayOfWeek
    except ValueError:
        print(f"Warning: Could not calculate features for date {year}-{month}-{day}")
        return None, None

# --- Prediction Worker Thread ---
def prediction_worker():
    global model, scaler, wr_scaler # Declare globals to modify them
    print(f"Prediction worker started. Using device: {device}")

    # Load model and scalers once
    print("Prediction worker loading model and scalers...")
    try:
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode

        scaler = joblib.load(SCALER_LOAD_PATH)
        wr_scaler = joblib.load(WR_SCALER_LOAD_PATH)
        print("Prediction worker: Model and scalers loaded successfully.")
        model_ready.set() # Signal that the model is ready
    except Exception as e:
        print(f"Prediction worker: Error loading model or scalers: {e}")
        # Model not ready, exit thread or handle recovery
        return

    while True:
        # Wait for a prediction request (blocking call)
        request = prediction_queue.get()
        if request is None: # Sentinel value to stop the thread
            break

        form_data = request['form_data']
        response_queue = request['response_queue'] # Queue to send results back to the client handler

        print(f"Prediction worker: Processing request for {form_data.get('submissionDate')}, Hour {form_data.get('submissionHour')}")

        try:
            year, month, day = map(int, form_data.get('submissionDate')[0].split('-'))
            startHour = int(form_data.get('submissionHour')[0])
            startDt = datetime(year, month, day, startHour)

            sequenceDataUnscaled = []
            successfulInput = True

            # Collect the 5 integer inputs (WR values)
            wr_values_input = []
            for i in range(1, 6):
                wr_values_input.append(float(form_data.get(f'integer{i}')[0]))

            # Populate sequenceDataUnscaled for the last 5 hours leading up to startHour
            # The form provides 5 WR values which are for the 5 hours *ending* at startHour
            # So, current_wr_index 0 is for (startHour - 4), index 4 is for startHour
            for i in range(SEQUENCE_LENGTH):
                currentDt = startDt - timedelta(hours=(SEQUENCE_LENGTH - 1 - i))
                currentYear = currentDt.year
                currentMonth = currentDt.month
                currentDay = currentDt.day
                currentHour = currentDt.hour

                weekOfYear, dayOfWeek = getDateFeatures(currentYear, currentMonth, currentDay)
                if weekOfYear is None:
                    successfulInput = False
                    break

                # The WR value from the form corresponds to this specific hour in the sequence
                wrValue = wr_values_input[i]

                sequenceDataUnscaled.append([
                    currentMonth, weekOfYear, currentDay,
                    currentHour, dayOfWeek, wrValue
                ])

            if not successfulInput:
                response_queue.put({"error": "Failed to gather sequence due to date calculation errors."})
                continue

            # --- Prepare and Scale the Initial Sequence ---
            initialSequenceUnscaled = np.array(sequenceDataUnscaled, dtype=np.float32)
            if initialSequenceUnscaled.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
                raise ValueError(f"Sequence data has wrong shape: {initialSequenceUnscaled.shape}")

            initialSequenceScaled = scaler.transform(initialSequenceUnscaled)

            predictedHours = []
            predictedWrs = []

            # Add the last input WR value and its hour to the predicted data for continuity
            predictedHours.append(startHour)
            predictedWrs.append(wr_values_input[-1]) # The last WR value provided in the form

            # Perform sequential prediction for the rest of the day (until hour 23)
            currentSequenceScaled = initialSequenceScaled.copy()

            with torch.no_grad():
                for predHourOffset in range(1, 24 - startHour): # From 1 hour after startHour up to 23
                    predDt = startDt + timedelta(hours=predHourOffset)
                    predYear = predDt.year
                    predMonth = predDt.month
                    predDay = predDt.day
                    predHour = predDt.hour

                    seqTensor = torch.FloatTensor([currentSequenceScaled]).to(device)
                    predictedWrScaled = model(seqTensor).item()
                    predictedWr = wr_scaler.inverse_transform(np.array([[predictedWrScaled]]))[0, 0]

                    predictedHours.append(predHour)
                    predictedWrs.append(predictedWr)

                    # Prepare features for the next prediction step
                    predWeek, predDotw = getDateFeatures(predYear, predMonth, predDay)
                    if predWeek is None:
                        print(f"Could not get features for {predYear}-{predMonth}-{predDay}. Stopping prediction.")
                        break

                    nextFeaturesActual = [predMonth, predWeek, predDay, predHour, predDotw]
                    nextStepInputUnscaled = np.append(nextFeaturesActual, predictedWr)
                    nextStepInputScaled = scaler.transform(nextStepInputUnscaled.reshape(1, -1)).flatten()
                    currentSequenceScaled = np.vstack((currentSequenceScaled[1:], nextStepInputScaled))

            graph_data = [{"hour": h, "value": wr} for h, wr in zip(predictedHours, predictedWrs)]

            response_queue.put({
                "message": "Prediction successful!",
                "submitted_date": form_data.get('submissionDate')[0],
                "submitted_hour": form_data.get('submissionHour')[0],
                "graph_data": graph_data
            })

        except Exception as e:
            print(f"Prediction worker: Error during prediction: {e}")
            response_queue.put({"error": f"Prediction failed: {e}"})


# --- Client Handler ---
def handle_client(conn, addr):
    print(f"Connected by {addr}")
    # Create a unique response queue for this client
    client_response_queue = queue.Queue()

    try:
        # Wait until the model is ready before processing requests
        model_ready.wait() # Block until model_ready.set() is called

        while True:
            request_bytes = conn.recv(4096)
            if not request_bytes:
                print(f"Client {addr} disconnected.")
                break

            request_data = request_bytes.decode('utf-8')
            print(f"Received from {addr}:\n{request_data}")

            status_code = "200 OK"
            content_type = "application/json"
            response_body = {}

            if request_data.startswith("POST"):
                print(f"Detected POST request from {addr}")
                body_start_index = request_data.find('\r\n\r\n')
                if body_start_index != -1:
                    post_body = request_data[body_start_index + 4:]
                    parsed_data = urllib.parse.parse_qs(post_body)
                    print(f"Parsed POST data: {parsed_data}")

                    # Put the request into the prediction queue
                    prediction_queue.put({
                        "form_data": parsed_data,
                        "response_queue": client_response_queue
                    })

                    # Wait for the prediction result (with a timeout)
                    try:
                        prediction_result = client_response_queue.get(timeout=60) # Wait up to 60 seconds
                        response_body = prediction_result
                        if "error" in prediction_result:
                            status_code = "500 Internal Server Error"
                    except queue.Empty:
                        status_code = "504 Gateway Timeout"
                        response_body = {"error": "Prediction timed out."}

                else:
                    status_code = "400 Bad Request"
                    response_body = {"error": "POST request body not found."}
            else:
                status_code = "405 Method Not Allowed"
                content_type = "text/plain"
                response_body = "Only POST requests are accepted."

            if content_type == "application/json":
                response_content = json.dumps(response_body)
            else:
                response_content = str(response_body)

            response_message = (
                f"HTTP/1.1 {status_code}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(response_content.encode('utf-8'))}\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                f"\r\n"
                f"{response_content}"
            )
            conn.sendall(response_message.encode('utf-8'))

    except ConnectionResetError:
        print(f"Client {addr} forcibly closed the connection.")
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

# --- Server Start Function ---
def start_server():
    # Start the prediction worker thread
    pred_thread = threading.Thread(target=prediction_worker, daemon=True)
    pred_thread.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            client_handler = threading.Thread(target=handle_client, args=(conn, addr))
            client_handler.start()

if __name__ == "__main__":
    start_server()
