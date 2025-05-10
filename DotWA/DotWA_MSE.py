# Colin Brown, April 6, CS4442: AI 2 Final Report
# DotWA_MSE.py Calculates the MSE of the DotWA model

import csv
import os
import math

# Config
PREDICTIONS_FILE = 'DotWA.csv' # input file containing pre-calculated average WR values
TEST_DATA_FILE = 'test_data.csv'   # input file with actual observed data for comparison

# Get full file paths
scriptDir = os.path.dirname(os.path.abspath(__file__))
predictionsPath = os.path.join(scriptDir, PREDICTIONS_FILE)
testDataPath = os.path.join(scriptDir, TEST_DATA_FILE)

# load the prediction data (average WR per month, day_of_week, hour)
predictions = []
with open(predictionsPath, 'r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        predictions.append([float(val) for val in row])

# Calc MSE 

# initialize variables for Mean Squared Error calculation
squaredErrorsSum = 0.0
dataPointCount = 0
skippedHoursCount = 0 # tracks points skipped by filters or missing predictions

# process the test data file row by row
with open(testDataPath, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader) # skip header row

    for i, row in enumerate(reader):
        # ensure row has enough columns before unpacking
        if len(row) < 6:
            print(f"Skipping row {i+2} in {TEST_DATA_FILE} due to insufficient columns: {row}")
            continue

        # extract relevant data fields from the test data row
        month = int(row[0])       # month (1-12)
        hour = int(row[3])        # hour (0-23)
        dotw = int(row[4])        # day of the week (1=Monday, 7=Sunday)
        observedWr = float(row[5]) # actual observed WR value

        predictionRowIndex = (month - 1) * 7 + (dotw - 1)
        predictionColIndex = hour

        predictedWr = predictions[predictionRowIndex][predictionColIndex]

        # skip calculation if the prediction is zero
        if predictedWr == 0:
            skippedHoursCount +=1
            continue

        # calculate the squared error for this data point
        error = observedWr - predictedWr
        squaredErrorsSum += error * error
        dataPointCount += 1

# calculate and print the final Mean Squared Error
if dataPointCount > 0:
    mse = squaredErrorsSum / dataPointCount
    print(f"\nProcessed {dataPointCount} data points.")
    if skippedHoursCount > 0:
        print(f"Skipped {skippedHoursCount} data points.")
    print(f"Mean Squared Error (MSE): {mse}")
else:
    print(f"\nError: No data points were processed successfully. Cannot calculate MSE.")
    if skippedHoursCount > 0:
        print(f"({skippedHoursCount} data points were skipped).")