# Colin Brown, April 6, CS4442: AI 2 Final Report
# transform_data.py First step of feature engineering tweet data. 

import csv
from datetime import datetime
import os

# Extract date and time information, add 0 WR vals, average complete duplicates, write to CSV 
def processCsv(inputFile, outputFile):
    dirPath = os.path.dirname(os.path.abspath(__file__)) # Get the directory
    with open(os.path.join(dirPath, inputFile), 'r') as inFile, open(os.path.join(dirPath, outputFile), 'w', newline='') as outFile:
        reader = csv.reader(inFile)
        writer = csv.writer(outFile)

        # Write the header row, including 'year'
        writer.writerow(['year', 'month', 'week', 'day', 'hour', 'dotW', 'WR'])

        data = []
        for row in reader: # read in tuples
            if len(row) == 1:
                row = row[0].split(',')
            if len(row) == 2: # extract features
                dateTimeStr, wr = row
                try:
                    dateTimeObj = datetime.strptime(dateTimeStr, '%Y-%m-%d %H:%M')
                    data.append((dateTimeObj, int(wr)))
                except ValueError:
                    print(f"Error processing row: {row}")

        processedData = []
        for dateTimeObj, wr in data: # get date features from date
            year = dateTimeObj.year
            month = dateTimeObj.month
            day = dateTimeObj.day
            dotw = dateTimeObj.isoweekday()
            hour = dateTimeObj.hour
            week = dateTimeObj.isocalendar()[1]
            processedData.append((year, month, day, dotw, hour, week, wr))

        # Prep to add zero values
        finalData = []
        dateGroups = {}
        for year, month, day, dotw, hour, week, wr in processedData:
            dateStr = f"{year}-{month}-{day}"
            if dateStr not in dateGroups:
                dateGroups[dateStr] = {}
            if hour not in dateGroups[dateStr]:
                dateGroups[dateStr][hour] = []
            dateGroups[dateStr][hour].append((wr, year, month, day, dotw, week))

        for dateStr, hourData in dateGroups.items():
            year = list(hourData.values())[0][0][1]
            month = list(hourData.values())[0][0][2]
            day = list(hourData.values())[0][0][3]
            dotw = list(hourData.values())[0][0][4]
            week = list(hourData.values())[0][0][5]

            hourValues = {}
            for hour, wrList in hourData.items():
                wrSum = sum(wr for wr, _, _, _, _, _ in wrList)
                wrAvg = wrSum / len(wrList)
                hourValues[hour] = wrAvg

            # Add zero values for morning and evenings if no value already
            if 5 <= month <= 8:  # Summer months (May-Aug)
                if 1 <= dotw <= 5:  # Weekdays (Mon-Fri)
                    for h in range(23, 24):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                    for h in range(0, 7):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))

                else:  # Weekends (Sat-Sun)
                    for h in range(17, 24):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                    for h in range(0, 9):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
            else:  # Sept-April
                if 1 <= dotw <= 4:  # Mon-Thurs
                    for h in range(23, 24):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                    for h in range(0, 7):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                elif dotw == 5: #Friday
                    for h in range(20, 24):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                    for h in range(0, 7):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                elif dotw == 6: #Saturday
                    for h in range(20, 24):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                    for h in range(0, 9):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                elif dotw == 7: #Sunday
                    for h in range(23, 24):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))
                    for h in range(0, 9):
                        if h not in hourValues:
                            finalData.append((year, month, week, day, h, dotw, 0))

            for hour, wrAvg in hourValues.items():
                finalData.append((year, month, week, day, hour, dotw, wrAvg))

        finalData.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4])) #Sort by year, month, week, day, hour.
        for year, month, week, day, hour, dotw, wr in finalData: # Write out tuples
            writer.writerow([year, month, week, day, hour, dotw, round(wr)]) 

# Example usage
inputCsvFile = 'raw_tweet_data.csv'
outputCsvFile = 'transformed_data.csv'
processCsv(inputCsvFile, outputCsvFile)