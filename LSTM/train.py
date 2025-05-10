# Colin Brown, April 6, CS4442: AI 2 Final Report
# train.py Trains a LSTM model, saving the one with the lowest validation MSE

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import random
from datetime import date
from sklearn.metrics import mean_squared_error

# import the model def
from model import LSTMModel

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, 'train_data.csv')
VALIDATION_DATA_PATH = os.path.join(SCRIPT_DIR, 'validation_data.csv')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'lstm_wr_model.pth')
SCALER_SAVE_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')
WR_SCALER_SAVE_PATH = os.path.join(SCRIPT_DIR, 'wr_scaler.joblib')
FEATURES = ['month', 'week', 'day', 'hour', 'DotW', 'WR'] # features used for training

# Hyperparameters
INPUT_SIZE = len(FEATURES) # number of input features
HIDDEN_SIZE = 64  # hidden layer size (neurons)
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 5 # number of previous time steps to use for prediction
BATCH_SIZE = 128    
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Generates sequences and corresponding target values from time series data.
def createSequences(data, sequenceLength):
    """
    Each sequence consists of 'sequenceLength' consecutive time steps,
    and target is 'WR' value of step immediately following the sequence.
    """
    xs, ys = [], []
    for i in range(len(data) - sequenceLength):
        x = data[i:(i + sequenceLength)]
        y = data[i + sequenceLength, -1] # target is the last column (WR)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

# Calculates the Mean Squared Error on the validation set by predicting sequences within each day starting from a random hour.
def calculateError(model, validDataGrouped, scaler, wrScaler, device):
    model.eval()
    totalDayError = 0
    numDays = 0
    allPredictions = []
    allActuals = []

    with torch.no_grad():
        for _, dayDf in validDataGrouped:
            dayDataOriginal = dayDf[['month', 'week', 'day', 'hour', 'DotW', 'WR']].values

            # Skip days with insufficient data for even one sequence
            if len(dayDataOriginal) <= SEQUENCE_LENGTH:
                print(f"Skipping day (too short): {dayDf['month'].iloc[0]}-{dayDf['day'].iloc[0]}")
                continue

            dayDataScaled = scaler.transform(dayDataOriginal) # Scale features using the loaded scaler

            # Determine the range for selecting a random prediction start point
            maxStartIndex = len(dayDataScaled) - SEQUENCE_LENGTH - 1
            if maxStartIndex < 0:
                # This case occurs if len(dayDataScaled) == SEQUENCE_LENGTH + 1 after the initial check
                print(f"Skipping day (too short after seq check): {dayDf['month'].iloc[0]}-{dayDf['day'].iloc[0]}")
                continue

            # Pick a random hour index to start prediction for the rest of the day
            # Ensure prediction starts at or after START_HOUR and leaves at least one point to predict
            startPredIndex = random.randint(SEQUENCE_LENGTH, len(dayDataScaled) - 1)

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
        print("No validation predictions were made across any days.")
        return float('inf')

    overallMse = mean_squared_error(allActuals, allPredictions)
    
    return overallMse if numDays > 0 else float('inf')

# Main training script
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")

    # load training and validation data
    print("Loading data...")
    trainDf = pd.read_csv(TRAIN_DATA_PATH)
    validationDf = pd.read_csv(VALIDATION_DATA_PATH)

    # select features and convert to numpy arrays
    trainData = trainDf[FEATURES].values
    # keep validation data separate for grouped processing later

    # scale training data and save scalers
    print("Scaling data...")
    # scale all features together using training data statistics
    scaler = MinMaxScaler()
    trainDataScaled = scaler.fit_transform(trainData)

    # create and fit a scaler specifically for the target variable (WR) for inverse transformation
    wrScaler = MinMaxScaler()
    wrScaler.fit(trainData[:, -1].reshape(-1, 1)) # fit only on the WR column

    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(wrScaler, WR_SCALER_SAVE_PATH)
    print(f"Scalers saved to {SCALER_SAVE_PATH} and {WR_SCALER_SAVE_PATH}")

    # create sequences from scaled training data
    print("Creating training sequences...")
    xTrain, yTrain = createSequences(trainDataScaled, SEQUENCE_LENGTH)

    # convert training data to pytorch tensors
    xTrainTensor = torch.FloatTensor(xTrain).to(device)
    yTrainTensor = torch.FloatTensor(yTrain).to(device)

    # create dataloader for efficient batch processing during training
    trainDataset = TensorDataset(xTrainTensor, yTrainTensor)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training data shape: X={xTrainTensor.shape}, y={yTrainTensor.shape}")

    # prepare validation data by grouping it by day
    # create a unique day identifier for grouping
    validationDf['day_id'] = validationDf['month'].astype(str) + '-' + validationDf['day'].astype(str)
    validationDataGrouped = validationDf.groupby('day_id')

    # initialize the LSTM model, loss function, and optimizer
    print("Initializing model...")
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss() # mse loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    bestValidationMse = float('inf') # track the best model based on validation error

    # training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train() # set model to training mode
        epochLoss = 0.0

        # iterate over training data batches
        for i, (batchX, batchY) in enumerate(trainLoader):
            # forward pass: compute model predictions
            outputs = model(batchX)
            loss = criterion(outputs, batchY)

            # backward pass: compute gradients and update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        averageEpochLoss = epochLoss / len(trainLoader)

        # validation phase
        # run validation multiple times for a more stable estimate due to random start point
        validationMseSum = 0
        numValidations = 5
        for _ in range(numValidations):
            validationMseSum += calculateError(model, validationDataGrouped, scaler, wrScaler, device)
        validationMse = validationMseSum / numValidations

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {averageEpochLoss:.6f}, Validation MSE: {validationMse:.4f}')

        # save the model if validation mse has improved
        if validationMse < bestValidationMse:
            bestValidationMse = validationMse
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"*** New best model saved with Validation MSE: {bestValidationMse:.4f} ***")

    print("Training finished.")
    # the model saved at MODEL_SAVE_PATH corresponds to the epoch with the lowest validation mse
    print(f"Best model saved to {MODEL_SAVE_PATH} with Validation MSE: {bestValidationMse:.4f}")

    # final confirmation that scalers are saved (already saved before the loop)
    print(f"Scalers available at {SCALER_SAVE_PATH} and {WR_SCALER_SAVE_PATH}")