import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from DataProcessor import DataProcessor

labelFontSize = 15
matplotlib.rc('font', size=labelFontSize)
matplotlib.rc('axes', titlesize=labelFontSize)

dataFolder = "data"
outputFolder = "Results"

i = 0

allDat = pd.DataFrame(
    columns=list(("Participant", "Average_Continuous_Stiffness", "Slope_Calculated_Stiffness"))
)

for file in os.listdir(dataFolder):
    patientData = DataProcessor(dataFolder, file, os.path.join(outputFolder, file[0:(len(file) - 4)]), fontSize=labelFontSize)
    stiffAvgCalc, stiffSlpCalc = patientData.FullPipline()

    allDat.loc[i] = [file[0:(len(file) - 4)], stiffAvgCalc, stiffSlpCalc]

    i += 1

indVals = allDat.index.values
AvgContVals = allDat.loc[:, "Average_Continuous_Stiffness"].values
SlpStiffVals = allDat.loc[:, "Slope_Calculated_Stiffness"].values

allDatPlot = plt.figure()
plt.clf()

plt.plot(indVals, AvgContVals, 'bo')
plt.plot(indVals, SlpStiffVals, 'ro')

plt.title("Participant Estimated Arm Stiffnesses")
plt.xlabel("Participant")
plt.ylabel("Calculated Stiffness (N/m)")
plt.legend(["Avg Method", "Reg Slope Method"])

allDatPlot.savefig(os.path.join(outputFolder, "allDataPlot.png"), bbox_inches='tight')

allDat.to_csv(os.path.join(outputFolder, "allData.csv"))
