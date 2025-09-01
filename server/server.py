# flask_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta

# --- New Imports for Tweet Scraping ---
import re
import pytz
from tweety import Twitter
from tweety.exceptions import RateLimitReached
from collections import defaultdict

# Import the LSTMModel from your local file
try:
    from model import LSTMModel
except ImportError:
    print("Error: model.py not found or LSTMModel class not defined. Please ensure 'model.py' is in the same directory.")
    exit(1)

# --- Flask App and LSTM Model Setup ---
app = Flask(__name__)
CORS(app)

# Twitter client sign in
app_twitter = Twitter("session")

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

# Tweet Scraping Endpoint 
@app.route('/gettweets', methods=['GET'])
def get_tweets():
    """
    Scrapes recent WR tweets from @WesternWeightRm, finds the 5-hour window
    leading up to the latest tweet, averages values for any hour with multiple
    tweets, and leaves hours with no tweets blank.
    """
    try:
        # Fetch a decent number of tweets to find the 5-hour sequence
        all_tweets = app_twitter.get_tweets("WesternWeightRm")
        timezone = pytz.timezone('America/Toronto')
        
        # Group tweet values by the hour they were posted
        tweets_by_hour = defaultdict(list)
        
        for tweet in all_tweets:
            tweetTxt = tweet.text.lower()
            wr_line = next((line for line in tweetTxt.splitlines() if "wr" in line), None)
            
            if not wr_line:
                continue

            found_num = re.findall(r"\d+", wr_line)
            if found_num:
                num = found_num[-1]
                
                # Get and convert timestamp
                tweetDate_str = re.findall(r".+:\d+:", str(tweet.date))[0][:-1]
                utc_datetime = datetime.strptime(tweetDate_str, '%Y-%m-%d %H:%M')
                local_datetime = utc_datetime.replace(tzinfo=pytz.utc).astimezone(timezone)
                
                # Use the hour as the key
                hour_key = local_datetime.replace(minute=0, second=0, microsecond=0)
                tweets_by_hour[hour_key].append(int(num))

        if not tweets_by_hour:
            return jsonify({"error": "Could not find any valid tweets."}), 500

        # Calculate the average value for each hour
        averaged_values = {hour: round(sum(vals) / len(vals)) for hour, vals in tweets_by_hour.items()}

        # Determine the most recent hour we have data for
        latest_hour = max(averaged_values.keys())

        # Build the final list for the 5 hours leading up to the latest hour
        final_values = []
        for i in range(5):
            # Check for data for each of the last 5 consecutive hours
            lookup_hour = latest_hour - timedelta(hours=(4 - i))
            value = averaged_values.get(lookup_hour)
            if value is None:
                if 0 <= lookup_hour.hour <= 6:
                    value = 0
                else:
                    value = ""
            final_values.append(value)
        
        response_data = {
            "date": latest_hour.strftime('%Y-%m-%d'),
            "hour": latest_hour.hour,
            "values": final_values
        }
        
        return jsonify(response_data)

    except RateLimitReached as e:
        print(f"Twitter rate limit reached: {e}")
        return jsonify({"error": f"Twitter rate limit reached. Please try again later."}), 429
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return jsonify({"error": f"An error occurred while fetching tweets: {e}"}), 500


# Prediction endpoint 
@app.route('/predict', methods=['POST'])
def make_prediction():
    """Endpoint for making a WR value prediction."""
    if not model:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        data = request.form
        year, month, day = map(int, data.get('submissionDate').split('-'))
        startHour = int(data.get('submissionHour'))
        startDt = datetime(year, month, day, startHour)
        
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
    load_model() # Load LSTM model
    app_twitter.connect() # Connect to twitter session
    #app.run(host='0.0.0.0', port=8080) # Server deployment
    app.run(host='127.0.0.1', port=8080) # Local testing