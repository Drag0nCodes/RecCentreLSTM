# Colin Brown, April 6, CS4442: AI 2 Final Report
# predict.py Askes for user input and makes a prediciton based on it with the LSTM model

import os
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt # import matplotlib

# import the model definition
from model import LSTMModel

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD_PATH = os.path.join(SCRIPT_DIR, 'lstm_wr_model.pth')
SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, 'scaler.joblib')
WR_SCALER_LOAD_PATH = os.path.join(SCRIPT_DIR, 'wr_scaler.joblib')

# Hyperparameters (must match training)
INPUT_SIZE = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 5

# calculates week of year and day of week (monday=1, sunday=7)
def getDateFeatures(year, month, day):
    try:
        dt = datetime(year, month, day)
        weekOfYear = dt.isocalendar()[1]
        dayOfWeek = dt.isoweekday() # monday is 1, sunday is 7
        return weekOfYear, dayOfWeek
    except ValueError:
        print(f"Warning: Could not calculate features for date {year}-{month}-{day}")
        return None, None


# --- Main Prediction Script ---
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set device (gpu or cpu)
    print(f"Using: {device}")

    # Load model and scalers
    print("Loading model and scalers...")
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device)) # load learned weights
    model.to(device) # move model to device
    model.eval() # set model to evaluation mode
    
    scaler = joblib.load(SCALER_LOAD_PATH)
    wrScaler = joblib.load(WR_SCALER_LOAD_PATH)

    print("Model and scalers loaded")


    # Get Input for start time
    print(f"\nEnter the date and hour for the *last known* WR value.")
    print(f"You will then be asked for the WR values for this hour and the previous {SEQUENCE_LENGTH - 1} hours.")
    while True:
        try:
            year = int(input("Enter Year (e.g., 2024): "))
            month = int(input("Enter Month (1-12): "))
            day = int(input("Enter Day (1-31): "))
            startHour = int(input(f"Enter the Hour (0-23) of the *last known* WR value: "))
            if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= startHour <= 23):
                raise ValueError("Invalid month, day, or hour.")
            startDt = datetime(year, month, day, startHour) # create starting datetime object
            break # exit loop if input is valid
        except ValueError as e:
            # print unchanged
            print(f"Invalid input: {e}. Please try again.")


    # Get user input for the WR sequence
    print(f"\nPlease enter the WR values for the following {SEQUENCE_LENGTH} time points:")
    sequenceDataUnscaled = []
    successfulInput = True
    for i in range(SEQUENCE_LENGTH - 1, -1, -1): # loop backwards to ask chronologically
        currentDt = startDt - timedelta(hours=i) # calculate datetime for this sequence point
        currentYear = currentDt.year
        currentMonth = currentDt.month
        currentDay = currentDt.day
        currentHour = currentDt.hour

        weekOfYear, dayOfWeek = getDateFeatures(currentYear, currentMonth, currentDay)
        if weekOfYear is None: # check if date features could be calculated
            successfulInput = False
            break

        while True: # loop until valid WR value is entered
            try:
                prompt = f"  - WR for {currentYear}-{currentMonth:02d}-{currentDay:02d} Hour {currentHour:02d}: "
                wrValue = float(input(prompt))
                # append all features for this time step
                sequenceDataUnscaled.append([
                    currentMonth, weekOfYear, currentDay,
                    currentHour, dayOfWeek, wrValue
                ])
                break # exit inner loop once valid WR is entered
            except ValueError:
                print("    Invalid input. Please enter a numeric WR value.")

        if not successfulInput: # exit outer loop if date features failed
              break

    if not successfulInput:
        print("Failed to gather sequence due to date calculation errors. Exiting.")
        exit()


    # --- Prepare and Scale the Initial Sequence ---
    try:
        initialSequenceUnscaled = np.array(sequenceDataUnscaled, dtype=np.float32) # convert list to numpy array
        if initialSequenceUnscaled.shape != (SEQUENCE_LENGTH, INPUT_SIZE): # validate shape
             raise ValueError(f"sequence data has wrong shape: {initialSequenceUnscaled.shape}")
        initialSequenceScaled = scaler.transform(initialSequenceUnscaled) # scale the features
    except Exception as e:
         print(f"Error occurred preparing the sequence: {e}")
         exit()


    # Store data for plotting 
    inputHours = initialSequenceUnscaled[:, 3].tolist() # extract hour column from input
    inputWrs = initialSequenceUnscaled[:, 5].tolist()   # extract WR column from input

    predictedHours = [] # list for predicted hours
    predictedWrs = []   # list for predicted WRs


    # Perform sequential prediction
    print(f"\nPredicting WR for {year}-{month:02d}-{day:02d} starting *after* hour {startHour}:")

    currentSequenceScaled = initialSequenceScaled.copy() # start with scaled input sequence

    with torch.no_grad(): # disable gradient calculation for prediction
        # loop from the hour after the last known hour until the end of the day (hour 23)
        for predHourOffset in range(1, 24 - startHour):
            predDt = startDt + timedelta(hours=predHourOffset) # calculate datetime for the prediction step
            predYear = predDt.year
            predMonth = predDt.month
            predDay = predDt.day
            predHour = predDt.hour

            seqTensor = torch.FloatTensor([currentSequenceScaled]).to(device) # prep tensor for the model
            predictedWrScaled = model(seqTensor).item() # get scaled prediction from model
            predictedWr = wrScaler.inverse_transform(np.array([[predictedWrScaled]]))[0, 0] # inverse scale prediction to get actual WR value

            print(f"Hour {predHour:02d}: Predicted WR = {predictedWr:.2f}")

            # store prediction for plotting
            predictedHours.append(predHour)
            predictedWrs.append(predictedWr)

            # prepare the features for the next prediction step
            predWeek, predDotw = getDateFeatures(predYear, predMonth, predDay)
            if predWeek is None: # stop if date features can't be calculated
                print(f"Could not get features for {predYear}-{predMonth}-{predDay}. Stopping prediction.")
                break

            
            nextFeaturesActual = [predMonth, predWeek, predDay, predHour, predDotw] # get known features for prediction hour
            nextStepInputUnscaled = np.append(nextFeaturesActual, predictedWr) # combine known features with predicted WR for this hour
            nextStepInputScaled = scaler.transform(nextStepInputUnscaled.reshape(1, -1)).flatten() # scale combined feature set
            currentSequenceScaled = np.vstack((currentSequenceScaled[1:], nextStepInputScaled)) # update the sequence: remove oldest step, add the new step

    print("\nPrediction finished.")


    # Plotting
    print("Generating plot...")
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(7, 5)) 

        # plot input data
        ax.plot(inputHours, inputWrs, marker='o', linestyle='-', color='blue', label=f'Input WR (Hours {min(inputHours):.0f}-{max(inputHours):.0f})')

        if predictedHours: # if predictions were made, plot
            ax.plot(predictedHours, predictedWrs, marker='x', linestyle='--', color='red', label=f'Predicted WR (Hours {min(predictedHours):.0f}-{max(predictedHours):.0f})')
        else:
             print("No predictions were made to plot.")

        # plot options
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("WR Value")
        ax.set_title(f"WR Input vs. Prediction for {year}-{month:02d}-{day:02d}")
        ax.legend() 
        ax.grid(True)

        # set x-axis ticks to show all relevant hours clearly
        allHours = sorted(list(set(inputHours + predictedHours)))
        if allHours:
            ax.set_xticks(range(int(min(allHours)), int(max(allHours)) + 1)) # ticks for the full relevant hour range
        # handle case where there might only be input hours (no predictions)
        elif inputHours:
             ax.set_xticks(range(int(min(inputHours)), int(max(inputHours)) + 1))

        plt.xticks(rotation=45) # rotate ticks slightly
        plt.tight_layout() # prevent labels overlapping
        plt.show()

    except Exception as e:
        print(f"\nError generating plot: {e}")
        print("Ensure matplotlib is installed ('pip install matplotlib') and data is valid.")