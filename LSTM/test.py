# Colin Brown, April 6, CS4442: AI 2 Final Report
# test.py tests the created LSTM model 50 times, averaging the MSE from all passes.

import os
import pandas as pd
import numpy as np
import torch
import joblib
import random
from datetime import date
from sklearn.metrics import mean_squared_error

# Import the model definition
from model import LSTMModel

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, 'test_data.csv')
MODEL_LOAD_PATH = os.path.join(SCRIPT_DIR, 'lstm_wr_model.pth')
SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')
WR_SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, 'wr_scaler.joblib')

# Model hyperparameters (must match training)
INPUT_SIZE = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 5 # Sequence length used during training
TEST_SIZE = 50      # Number of test passes to average MSE over
START_HOUR = 5      # Earliest hour index to start predicting from (must be >= SEQUENCE_LENGTH)


# Calculates the Mean Squared Error on the test set by predicting sequences within each day starting from a random hour.
def calculateError(model, testDataGrouped, scaler, wrScaler, device):
    model.eval()
    totalDayError = 0
    numDays = 0
    allPredictions = []
    allActuals = []

    with torch.no_grad():
        for _, dayDf in testDataGrouped:
            dayDataOriginal = dayDf[['month', 'week', 'day', 'hour', 'DotW', 'WR']].values

            # Skip days with insufficient data for even one sequence
            if len(dayDataOriginal) <= SEQUENCE_LENGTH:
                print(f"Skipping day in test (too short): {dayDf['month'].iloc[0]}-{dayDf['day'].iloc[0]}")
                continue

            dayDataScaled = scaler.transform(dayDataOriginal) # Scale features using the loaded scaler

            # Determine the range for selecting a random prediction start point
            maxStartIndex = len(dayDataScaled) - SEQUENCE_LENGTH - 1
            if maxStartIndex < 0:
                # This case occurs if len(dayDataScaled) == SEQUENCE_LENGTH + 1 after the initial check
                print(f"Skipping day in test (too short after seq check): {dayDf['month'].iloc[0]}-{dayDf['day'].iloc[0]}")
                continue

            # Pick a random hour index to start prediction for the rest of the day
            # Ensure prediction starts at or after START_HOUR and leaves at least one point to predict
            startPredIndex = random.randint(START_HOUR, len(dayDataScaled) - 1)

            # Sequential Prediction
            # Extract the initial sequence leading up to the first prediction point
            currentSequenceScaled = dayDataScaled[startPredIndex - SEQUENCE_LENGTH : startPredIndex].copy()
            dayPredictionsScaled = []
            dayActualsScaled = [] # Store corresponding actual scaled values

            # Predict hour by hour from the start index to the end of the day
            for i in range(startPredIndex, len(dayDataScaled)):
                # Prepare input tensor (batch size 1)
                seqTensor = torch.FloatTensor([currentSequenceScaled]).to(device) # Shape (1, seq_len, input_size)

                # Predict the next scaled WR value
                predictedWrScaled = model(seqTensor).item()
                dayPredictionsScaled.append(predictedWrScaled)

                # Record the actual scaled WR value for this time step
                actualWrScaled = dayDataScaled[i, -1]
                dayActualsScaled.append(actualWrScaled)

                # Prepare the input sequence for the *next* time step:
                # Use actual features from the current time step...
                nextFeaturesActual = dayDataScaled[i, :-1]
                # ...but append the predicted WR for this step.
                nextStepInputScaled = np.append(nextFeaturesActual, predictedWrScaled)
                currentSequenceScaled = np.vstack((currentSequenceScaled[1:], nextStepInputScaled)) # Roll the sequence window forward

            # Inverse transform the scaled predictions and actuals to original WR scale
            predictionsOriginal = wrScaler.inverse_transform(np.array(dayPredictionsScaled).reshape(-1, 1))
            actualOriginal = wrScaler.inverse_transform(np.array(dayActualsScaled).reshape(-1, 1))

            # Store results for overall MSE calculation
            allPredictions.extend(predictionsOriginal.flatten().tolist())
            allActuals.extend(actualOriginal.flatten().tolist())
            numDays +=1

    # Calculate overall MSE across all predicted points from all processed days
    if not allActuals:
        print("No test predictions were made across any days.")
        return float('inf')

    overallMse = mean_squared_error(allActuals, allPredictions)
    print(f"MSE over {len(allActuals)} data points from {numDays} days: {overallMse:.4f}")
    return overallMse


# Main test script part
# Set computation device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and scalers
print("Loading model and scalers...")
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE) # Instantiate model structure
model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device)) # Load learned weights
model.to(device)
model.eval() # Set model to eval mode

# Load the scalers used during training
scaler = joblib.load(SCALER_LOAD_PATH)
wrScaler = joblib.load(WR_SCALER_LOAD_PATH)

print("Model and scalers loaded.")


# --- Load test data ---
print(f"Loading test data from {TEST_DATA_PATH}...")
testDf = pd.read_csv(TEST_DATA_PATH)
# Verify necessary columns are present
requiredCols = ['month', 'week', 'day', 'hour', 'DotW', 'WR']

# Prepare test data
# Create a unique identifier for each day to group by
testDf['dayId'] = testDf['month'].astype(str) + '-' + testDf['day'].astype(str)
testDataGrouped = testDf.groupby('dayId')
print(f"Test data contains {len(testDataGrouped)} unique days.")

# Calculate average test MSE
print(f"Calculating test MSE over {TEST_SIZE} passes...")
mseSum = 0
for i in range(TEST_SIZE):
    currentMse = calculateError(model, testDataGrouped, scaler, wrScaler, device)
    mseSum += currentMse
aveMse = mseSum / TEST_SIZE # Calc ave MSE

print(f"Overall average test mean squared error (MSE): {aveMse:.4f}")
