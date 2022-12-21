import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.linear_model import LinearRegression

class DataProcessor():
    def __init__(self, folderLocation, filename, outputDestination=".", fontSize=22):
        self.indArr = []
        self.forceArr = []
        self.offsetArr = []
        self.stiffArr = []
        self.regForces = []
        self.confIntUp = []
        self.confIntDn = []
        self.avgStiffness = 0
        self.stiffSlope = 0
        self.dataPath = os.path.join(folderLocation, filename)
        self.outputDir = outputDestination
        self.patient = filename[0:(len(filename) - 4)]

        matplotlib.rc('font', size=fontSize)
        matplotlib.rc('axes', titlesize=fontSize)

    def LoadData(self):
        dataFrame = pd.read_csv(self.dataPath)
        dataArr = dataFrame.to_numpy()
        indDat = []

        for ind in dataArr[:, 0]:
            indDat.append(int(ind - dataArr[0, 0]))

        self.indArr = np.array(indDat)
        self.forceArr = dataArr[:, 1]
        self.offsetArr = np.abs(dataArr[:, 2])


    def CalcConStiffness(self):
        for i in self.indArr:
            stiff = self.forceArr[i] / self.offsetArr[i]
            self.stiffArr.append(stiff)

        self.avgStiffness = np.mean(self.stiffArr)

    def PlotForceVsOffset(self):
        fig = plt.figure()
        plt.clf()

        plt.scatter(self.offsetArr, self.forceArr, s=0.5)
        plt.plot(self.offsetArr, self.regForces, 'r-')
        plt.plot(self.offsetArr, self.confIntUp, 'k--')
        plt.plot(self.offsetArr, self.confIntDn, 'k--')
        plt.title("Force over Displacement")
        plt.xlabel("Offset (m)")
        plt.ylabel("Force (N)")
        plt.legend(["Collected Data", "LR fit line", "Conf Interval"])

        return fig


    def PlotStiffVsIndex(self):
        fig = plt.figure()
        plt.clf()

        plt.plot(self.indArr, self.stiffArr)
        plt.title("Realtime Stiffness over Press")
        plt.xlabel("Press Index")
        plt.ylabel("Stiffness (N/m)")
        
        return fig

    def FvOffRegression(self):
        model = LinearRegression()

        model.fit(self.offsetArr.reshape(-1, 1), self.forceArr)

        self.regForces = model.predict(self.offsetArr.reshape(-1, 1))

        forceDiff = self.regForces[2] - self.regForces[1]
        offDiff = self.offsetArr[2] - self.offsetArr[1]

        self.stiffSlope = forceDiff/offDiff

    def CalcConfInterval(self, perInt=0.95):
        sumErr = np.sum((self.forceArr - self.regForces)**2)
        stdev = np.sqrt(1 / (len(self.forceArr) - 2) * sumErr)

        oneMinPi = 1 - perInt
        ppfLookup = 1 - (oneMinPi / 2)
        zScore = st.norm.ppf(ppfLookup)

        interval = zScore * stdev

        lower, upper = self.regForces - interval, self.regForces + interval

        self.confIntUp = upper
        self.confIntDn = lower

    def MakenStoreFigures(self):
        FvOfig = self.PlotForceVsOffset()
        SvIfig = self.PlotStiffVsIndex()

        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        outPath = os.path.join(self.outputDir, self.patient)

        FvOfig.savefig(outPath + "fvo" + ".png")
        SvIfig.savefig(outPath + "svi" + ".png", bbox_inches='tight')

    def FullPipline(self):
        self.LoadData()
        self.CalcConStiffness()
        self.FvOffRegression()
        self.CalcConfInterval()
        self.MakenStoreFigures()

        return [self.avgStiffness, self.stiffSlope]

    def PrintArr(self):

        if self.offsetArr.count == 0:
            print("Data array of object is null and cannot be printed")
            return

        print(self.offsetArr)
        print(len(self.offsetArr))

    def PrintInfo(self):
        print(self.avgStiffness)



# self.confInt = st.t.interval(alpha=perInt, df=len(self.regForces)-1, loc=np.mean(self.regForces), scale=st.sem(self.regForces))
