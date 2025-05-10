# Colin Brown, April 6, CS4442: AI 2 Final Report
# create_train_valid_test_interpolate.py Second step of feature engineering tweet data. 

import os
import pandas as pd
import random
import numpy as np
import traceback

# Fills missing hours (0-23) within each day of a DataFrame with interpolation
def fillDailyGaps(inputDf, fillValueNa=0):
    # return an empty DataFrame with the expected columns if input is empty
    if inputDf.empty:
        return pd.DataFrame(columns=['month', 'week', 'day', 'hour', 'dotW', 'WR'])

    print(f"  Applying gap filling to DataFrame with {len(inputDf)} rows...")
    filledDataList = []
    requiredCols = ['month', 'week', 'day', 'hour', 'dotW', 'WR'] # ensure 'dotW' matches case used in df

    # check if required columns are present
    if not all(col in inputDf.columns for col in requiredCols):
        print("  Error: Input DataFrame missing one or more required columns for gap filling.")
        print(f"  Expected: {requiredCols}, Found: {list(inputDf.columns)}")
        # attempt to return a structure consistent with expectations if possible
        return pd.DataFrame(columns=requiredCols)

    # identify unique days to process individually
    uniqueDayIdentifiers = inputDf[['month', 'day']].drop_duplicates().values.tolist()

    for m, d in uniqueDayIdentifiers:
        # get data for the specific day
        dayDf = inputDf[(inputDf['month'] == m) & (inputDf['day'] == d)].copy()
        if dayDf.empty: continue

        # handle rows with the same hour within the day by averaging WR
        if dayDf.duplicated(subset=['hour']).any():
              print(f"    Found duplicate hours for month={m}, day={d}. Averaging WR.")
              # define how to aggregate each column when duplicates exist for an hour
              aggregationFunctions = {
                  'WR': 'mean',      # average WR
                  'month': 'first',   # should be constant for the day
                  'day': 'first',     # should be constant for the day
                  'week': 'first',    # take the first week value encountered
                  'dotW': 'first'     # take the first dotW value encountered
              }
              # ensure all required columns are considered for aggregation
              columnsToAggregate = {col: aggregationFunctions[col] for col in requiredCols if col in dayDf.columns and col != 'hour'}

              dayDf = dayDf.groupby('hour', as_index=False).agg(columnsToAggregate)

              # round WR after averaging
              if 'WR' in dayDf.columns:
                  # keep as float temporarily in case interpolation introduces NaNs
                  dayDf['WR'] = dayDf['WR'].round()

        # sort by hour to ensure correct interpolation/filling later
        dayDf = dayDf.sort_values(by='hour')

        # prepare for reindexing to ensure all hours are present
        fullHourIndex = pd.Index(range(24), name='hour')
        # select only the necessary columns for processing
        columnsToProcess = [col for col in requiredCols if col != 'hour' and col in dayDf.columns]

        # check if essential columns exist after potential aggregation
        if not all(c in dayDf.columns for c in ['hour'] + columnsToProcess):
            print(f"    Skipping day {m}-{d} due to missing columns after aggregation.")
            continue

        # set index to hour for reindexing and interpolation
        dayDfIndexed = dayDf.set_index('hour')[columnsToProcess]

        # ensure all hours (0-23) are present for the day
        dayDfReindexed = dayDfIndexed.reindex(fullHourIndex)

        # interpolate missing WR values linearly
        if 'WR' in dayDfReindexed.columns:
            # interpolate first, limit ensures filling start/end gaps if possible
            dayDfReindexed['WR'] = dayDfReindexed['WR'].interpolate(method='linear', limit_direction='both')
            dayDfReindexed['WR'] = dayDfReindexed['WR'].round() # round after interpolation
            dayDfReindexed['WR'].fillna(fillValueNa, inplace=True) # fill any remaining NaNs
        else:
             # handle case where WR column might have been missing entirely
             dayDfReindexed['WR'] = fillValueNa

        # fill month and day using the day's unique identifiers (they are constant for the day)
        dayDfReindexed['month'] = m
        dayDfReindexed['day'] = d

        # fill other potentially missing columns ('week', 'dotW') using ffill/bfill
        fillColumnsFfill = ['week', 'dotW']
        for col in fillColumnsFfill:
            if col in dayDfReindexed.columns:
                # fill using previous or next valid observation within the day
                dayDfReindexed[col] = dayDfReindexed[col].ffill().bfill()
                # final fallback fill if the entire day was missing this value initially
                dayDfReindexed[col].fillna(fillValueNa, inplace=True)
            else:
                 # if column didn't exist (e.g., missing from inputDf or aggregation)
                 dayDfReindexed[col] = fillValueNa

        # convert specified columns back to integer type
        # Note: ensure WR is filled before converting to int
        integerColumns = ['month', 'day', 'week', 'dotW']
        if 'WR' in dayDfReindexed.columns:
            integerColumns.append('WR')

        for col in integerColumns:
            if col in dayDfReindexed.columns:
                # ensure no NaNs before converting to int, using the defined fill value
                dayDfReindexed[col] = dayDfReindexed[col].fillna(fillValueNa).round().astype(int)

        # convert index 'hour' back to a column
        filledDayDf = dayDfReindexed.reset_index()

        # ensure the final dataframe has the correct columns in the standard order
        for col in requiredCols:
             if col not in filledDayDf.columns:
                  filledDayDf[col] = fillValueNa # add missing columns with default value
        filledDayDf = filledDayDf[requiredCols] # reorder/select final columns

        filledDataList.append(filledDayDf)

    # combine data from all processed days
    if not filledDataList:
        print("  Warning: No data generated after gap filling.")
        return pd.DataFrame(columns=requiredCols) # return empty df with standard columns

    finalFilledDf = pd.concat(filledDataList, ignore_index=True)

    print(f"  Gap filling resulted in DataFrame with {len(finalFilledDf)} rows.")
    return finalFilledDf

# Average partial dublicates, split data into train, validation, and test sets
def createTrainTestValidationData(inputFile='transformed_data.csv',
                                  testFile='test_data.csv',
                                  trainFile='train_data.csv',
                                  validationFile='validation_data.csv',
                                  testPercentage=0.1,
                                  validationPercentage=0.1):
# determine script directory and input file path
    dirPath = os.path.dirname(os.path.abspath(__file__))
    inputPath = os.path.join(dirPath, inputFile)
    dfInitial = pd.read_csv(inputPath)
    print(f"Read {len(dfInitial)} rows from {inputPath}")

    # average WR for entries with the same day/hour combination across the dataset
    # assumes 'DotW' should be 'dotW' if consistency is needed
    if 'DotW' in dfInitial.columns and 'dotW' not in dfInitial.columns:
        dfInitial.rename(columns={'DotW': 'dotW'}, inplace=True) # rename for consistency
    print("Averaging WR for duplicate day/hour entries...")
    # define columns that uniquely identify a time point for averaging WR
    groupColumns = ['month', 'week', 'day', 'hour', 'dotW']
    dfAveraged = dfInitial.groupby(groupColumns, as_index=False)['WR'].mean()
    dfAveraged['WR'] = dfAveraged['WR'].round().astype(int) # round and ensure integer WR
    print(f"Data reduced to {len(dfAveraged)} rows after averaging.")

    # Split Data by Day (before gap filling)
    print("\nSplitting data into train, validation, and test sets by day...")
    # identify unique days in the dataset for splitting
    uniqueDays = dfAveraged[['month', 'week', 'dotW', 'day']].drop_duplicates()
    uniqueDaysList = list(uniqueDays.itertuples(index=False, name=None))

    if not uniqueDaysList:
        raise ValueError("Could not find unique days in the averaged data.")

    # calculate number of days for each set
    numTotalDays = len(uniqueDaysList)
    numTestDays = int(numTotalDays * testPercentage)
    numValidationDays = int(numTotalDays * validationPercentage)
    # ensure counts don't exceed available days
    numTestDays = min(numTestDays, numTotalDays)
    numValidationDays = min(numValidationDays, numTotalDays - numTestDays)
    numTrainDays = numTotalDays - numTestDays - numValidationDays

    print(f"Total unique days: {numTotalDays}")
    print(f"Selecting {numTestDays} days for test set.")
    print(f"Selecting {numValidationDays} days for validation set.")
    print(f"Using remaining {numTrainDays} days for training set.")

    # shuffle days randomly for unbiased splitting
    random.shuffle(uniqueDaysList)
    testDaysSet = set(uniqueDaysList[:numTestDays])
    validationDaysSet = set(uniqueDaysList[numTestDays : numTestDays + numValidationDays])
    trainDaysSet = set(uniqueDaysList[numTestDays + numValidationDays:])

    # create initial train/validation/test splits based on the selected days
    # use a temporary tuple column for efficient filtering
    dfAveraged['day_tuple'] = dfAveraged[['month', 'week', 'dotW', 'day']].apply(tuple, axis=1)

    testDfInitial = dfAveraged[dfAveraged['day_tuple'].isin(testDaysSet)].drop(columns=['day_tuple']).copy()
    validationDfInitial = dfAveraged[dfAveraged['day_tuple'].isin(validationDaysSet)].drop(columns=['day_tuple']).copy()
    # training data does not get gap-filled, so it's final here
    trainDfFinal = dfAveraged[dfAveraged['day_tuple'].isin(trainDaysSet)].drop(columns=['day_tuple']).copy()

    print(f"Initial split rows: Train={len(trainDfFinal)}, Validation={len(validationDfInitial)}, Test={len(testDfInitial)}.")

    # Apply Gap Filling ONLY to Validation and Test Data ---
    print("\nApplying gap filling to Validation set...")
    validationDfFinal = fillDailyGaps(validationDfInitial)

    print("\nApplying gap filling to Test set...")
    testDfFinal = fillDailyGaps(testDfInitial)

    # Sort and save the final dataFrames
    print("\nSorting final datasets...")
    # sort final dataframes by date/time for consistency
    sortOrder = ['month', 'day', 'dotW', 'hour']
    trainDfFinal = trainDfFinal.sort_values(by=sortOrder).reset_index(drop=True)
    validationDfFinal = validationDfFinal.sort_values(by=sortOrder).reset_index(drop=True)
    testDfFinal = testDfFinal.sort_values(by=sortOrder).reset_index(drop=True)

    # save the processed datasets to CSV files
    trainPath = os.path.join(dirPath, trainFile)
    trainDfFinal.to_csv(trainPath, index=False)
    print(f"Training data (NO gap fill) ({len(trainDfFinal)} rows) saved to {trainPath}")

    validationPath = os.path.join(dirPath, validationFile)
    validationDfFinal.to_csv(validationPath, index=False)
    print(f"Validation data (gaps filled) ({len(validationDfFinal)} rows) saved to {validationPath}")

    testPath = os.path.join(dirPath, testFile)
    testDfFinal.to_csv(testPath, index=False)
    print(f"Test data (gaps filled) ({len(testDfFinal)} rows) saved to {testPath}")


if __name__ == "__main__":
    # execute the data splitting and processing function
    createTrainTestValidationData()