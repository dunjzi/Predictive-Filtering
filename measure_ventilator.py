import pandas as pd
import time
import math
import padasip as pa
import numpy as np
import adaptfilt
import statsmodels.tsa.api as smt
import warnings
warnings.filterwarnings('ignore')
from rfidutil import *
from sensor import *
from measure import *
from matplotlib import cm, pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from measure_filterprx import FilterprxMeasure
from sklearn.metrics import mean_squared_error
import scipy.stats
import scipy.integrate as integrate



class VentilatorMeasure (Measure): 
    def __init__(self, _sensor, _deltat=0.02, _gobackn=20, _noisePeakTime = 0.3): #_noisePeakTime = 0.5
        super(VentilatorMeasure , self).__init__(_sensor)
        self.deltat = _deltat
        self.noisePeaks = _noisePeakTime
        self.gobackn = _gobackn # in seconds
        self.stepsOfPercentError = [175, 150, 125, 100, 75, 50, 25, 0]
        self.voterInfo = {175: [], 150: [], 125: [], 100: [], 75: [], 50: [], 25: [], 0: []}
        self.title = 'K-means: Reduce Noise, Classification, and Prediction'
        self.peakDifference = []
        self.peakStartConstant = []
        self.size = 0
        self.estprevrrinsec = float(30/60)
        self.predictModel_AP_Single = pa.filters.FilterAP(1, mu=1.)
        self.predictModel_GNGD_Single = pa.filters.FilterGNGD(1, mu=1.)
        self.predictModel_NLMS_Single = pa.filters.FilterNLMS(1, mu=1.)
        self.predictModel_AP_Multiple = pa.filters.FilterAP(3, mu=1.)
        self.predictModel_GNGD_Multiple = pa.filters.FilterGNGD(3, mu=1.)
        self.predictModel_NLMS_Multiple = pa.filters.FilterNLMS(3, mu=1.)
        
    def noiseReduction(self, data, relative_timestamp):
        if(len(data) >= 18): #To apply Sayandeep's filter, we need to have a minimum of 18 data points before filtering
            m = FilterprxMeasure(self)
            data = m.filterPrx(data) #Sayandeep's filter
            for i in self.stepsOfPercentError:
                squareWave, stretchingDataRepresentedBy = self.kMeansModel(data, i, relative_timestamp)
                if(stretchingDataRepresentedBy == -1):
                    print "Error, check device! Unable to determine breathing data before noise reduction process!"
                    return 
                else:
                    squareWave = self.identifyNoiseWidths(squareWave, stretchingDataRepresentedBy)
                    if(i == 0):
                        self.findWidthStartAndEnd(squareWave, i)
                        squareWave = self.votingProcess(squareWave)
                        new_data = []
                        self.voterInfo = {175: [], 150: [], 125: [], 100: [], 75: [], 50: [], 25: [], 0: []}
                        squareWave = self.finalCheckOnKmeans(squareWave, data)
                        squareWave = self.LastCheck(squareWave)
                        return squareWave
                    else:
                        self.findWidthStartAndEnd(squareWave, i)
                        
                        
    def kMeansModel(self, data, acceptedPercentDifference, relative_timestamp):
        model = KMeans(n_clusters = 2)
        newData = []
        for i in data:
            temp = []
            temp.append(i)
            newData.append(temp)
        model = model.fit(newData)
        modelTransform = model.transform(newData) #Get the distance from each data point to both centroids
        modelPrediction = list(model.predict(newData))
        squareWave = modelPrediction
        stretchingDataRepresentedBy = self.determineStretchingData(modelPrediction, data)        
        modelPrediction = self.checkKmeansClassification(modelTransform, acceptedPercentDifference, stretchingDataRepresentedBy, modelPrediction)
        ###############################
        for i in range(0,len(modelPrediction)):
            if(stretchingDataRepresentedBy == 0):
                if(modelPrediction[i] == 0):
                    modelPrediction[i] = 1
                else:
                    modelPrediction[i] = 0
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        if(len(data) != len(relative_timestamp)):
            removethis = []
            removethis.append(len(data) - 1)
            new_data = np.delete(data,removethis)
            squareWave.pop()
            plt.subplot(2, 1, 1)
            plt.plot(relative_timestamp,new_data)
            plt.title("prx over time for 20 secs of normal breathing")
            plt.ylabel('filtered prx (dbm)') 
            plt.xlabel('time (seconds)')
            plt.subplot(2, 1, 2)
            plt.plot(relative_timestamp,squareWave)
            plt.title("square wave over time for 20 secs of normal breathing")
            plt.ylabel('classification square wave') 
            plt.xlabel('time (seconds)')
        else:
            plt.subplot(2, 1, 1)
            plt.plot(relative_timestamp,data)
            plt.title("prx over time for 20 secs of normal breathing")
            plt.ylabel('filtered prx (dbm)') 
            plt.xlabel('time (seconds)')
            plt.subplot(2, 1, 2)
            plt.plot(relative_timestamp,squareWave)
            plt.title("square wave over time for 20 secs of normal breathing")
            plt.ylabel('classification square wave') 
            plt.xlabel('time (seconds)')
        plt.show()
        ###############################
        return modelPrediction, stretchingDataRepresentedBy

    def percentDifference(self, valueOne, valueTwo):
        return (float(math.fabs(valueOne - valueTwo))/(float(valueOne + valueTwo)/2)) * 100

    def checkKmeansClassification(self, modelTransform, acceptedPercentDifference, stretchingDataRepresentedBy, modelPrediction):
        misclassifiedData = []
        for i in range(0,len(modelTransform)):
            if(stretchingDataRepresentedBy != modelPrediction[i]):
                if(self.percentDifference(modelTransform[i][0], modelTransform[i][1]) < float(acceptedPercentDifference)): #Might be misclassified
                    if(stretchingDataRepresentedBy == 1):
                        modelPrediction[i] = 1
                    else:
                        modelPrediction[i] = 0
        return modelPrediction

    def determineStretchingData(self, squareWave, data):
        maxInList = data[0]
        maxPos = 0
        minInList = data[0]
        minPos = 0
        for i in range(0,len(data)):
            if(data[i] > maxInList):
                maxInList = data[i]
                maxPos = i
            if(data[i] < minInList):
                minInList = data[i]
                minPos = i
        kMeansMaxClass = squareWave[maxPos]
        kMeansMinClass = squareWave[minPos]
        if(kMeansMaxClass == kMeansMinClass):
            return -1
        elif(kMeansMaxClass == 1):
            return 1
        else:
            return 0

    def identifyNoiseWidths(self, squareWave, stretchingDataRepresentedBy):
        allPeakWidths, allDipWidths = self.findWidth(squareWave)
        if(not((len(allPeakWidths) == 0) or (len(allDipWidths) == 0))):
            maxValuePeaks = allPeakWidths[0]
            maxValueDips = allDipWidths[0]
            maxPos = 0
            model = KMeans(n_clusters = 2)
            stretchingWidths = []
            nonStretchingWidths = []
            if(stretchingDataRepresentedBy == 1):
                for i in range(0,len(allPeakWidths)):
                    temp = []
                    temp.append(allPeakWidths[i])
                    stretchingWidths.append(temp)
                    if(allPeakWidths[i] > maxValuePeaks):
                        maxValuePeaks = allPeakWidths[i]
                        maxPos = i
                nonStretchingWidths = allDipWidths
            else:
                for i in range(0,len(allDipWidths)):
                    temp = []
                    temp.append(allDipWidths[i])
                    stretchingWidths.append(temp)
                    if(allDipWidths[i] > maxValueDips):
                        maxValueDips = allDipWidths[i]
                        maxPos = i
                nonStretchingWidths = allPeakWidths
            model = model.fit(stretchingWidths)
            modelPrediction = model.predict(stretchingWidths)
            validStretchDataRepresentedBy = modelPrediction[maxPos]
            return self.removeNoise(validStretchDataRepresentedBy, stretchingDataRepresentedBy, modelPrediction, stretchingWidths, nonStretchingWidths, squareWave[0])
        else:
            return squareWave

    def findWidth(self, squareWave):
        allPeaksWidth = []
        allDipsWidth = []
        widthOfPeak = 0
        widthOfDip = 0
        for i in range(0, len(squareWave)):
            if(squareWave[i] == 1):
                widthOfPeak += 1
            if(squareWave[i] == 1 and (i+1 >= len(squareWave) or squareWave[i+1] == 0)):
                allPeaksWidth.append(widthOfPeak)
                widthOfPeak = 0
            if(squareWave[i] == 0):
                widthOfDip +=1
            if(squareWave[i] == 0 and (i+1 >= len(squareWave) or squareWave[i+1] == 1)):
                allDipsWidth.append(widthOfDip)
                widthOfDip = 0
        return allPeaksWidth, allDipsWidth 

    def removeNoise(self, validStretchDataRepresentedBy, stretchingDataRepresentedBy, classification, stretchingWidths, nonStretchingWidths, startingValue):
        temp = []
        for i in stretchingWidths:
            temp.append(i[0])
        stretchingWidths = temp 
        
        newSquareWave = []
        stillMerging = True
        countStretch = 0
        countNonStretch = 0
        stretchSize = len(stretchingWidths) - 1
        nonStretchSize = len(nonStretchingWidths) - 1
        if(stretchingDataRepresentedBy == startingValue):
            while(stillMerging):
                if(countStretch <= stretchSize):
                    if(not(validStretchDataRepresentedBy == classification[countStretch])):
                        for i in range(0,stretchingWidths[countStretch]):
                            newSquareWave.append(0)
                    else:
                        for i in range(0,stretchingWidths[countStretch]):
                            newSquareWave.append(1)
                    countStretch = countStretch + 1
                if(countNonStretch <= nonStretchSize):
                    for i in range(0,nonStretchingWidths[countNonStretch]):
                        newSquareWave.append(0)
                    countNonStretch = countNonStretch + 1
                if((countNonStretch + countStretch) == (nonStretchSize + stretchSize + 2)):
                    stillMerging = False
        else:
            while(stillMerging):
                if(countNonStretch <= nonStretchSize):
                    for i in range(0,nonStretchingWidths[countNonStretch]):
                        newSquareWave.append(0)
                    countNonStretch = countNonStretch + 1
                if(countStretch <= stretchSize):
                    if(not(validStretchDataRepresentedBy == classification[countStretch])):
                        for i in range(0,stretchingWidths[countStretch]):
                            newSquareWave.append(0)
                    else:
                        for i in range(0,stretchingWidths[countStretch]):
                            newSquareWave.append(1)
                    countStretch = countStretch + 1
                if((countNonStretch + countStretch) == (nonStretchSize + stretchSize + 2)):
                    stillMerging = False
        return newSquareWave

    def findWidthStartAndEnd(self, squareWave, currentAcceptedPercentError):
        peakInfo = ['p']
        dipInfo = ['d']
        for i in range(0, len(squareWave)):
            if((squareWave[i] == 1) and (((i - 1 >= 0) and (squareWave[i-1] == 0)) or i == 0)):
                peakInfo.append(i)
            if(squareWave[i] == 1 and (i+1 >= len(squareWave) or squareWave[i+1] == 0)):
                peakInfo.append(i)
            if((squareWave[i] == 0) and (((i - 1 >= 0) and (squareWave[i-1] == 1)) or i == 0)):
                dipInfo.append(i)
            if(squareWave[i] == 0 and (i+1 >= len(squareWave) or squareWave[i+1] == 1)):
                dipInfo.append(i)
            if(len(peakInfo) == 3):
                self.voterInfo[currentAcceptedPercentError].append(peakInfo)
                peakInfo = ['p']
            if(len(dipInfo) == 3):
                self.voterInfo[currentAcceptedPercentError].append(dipInfo)
                dipInfo = ['d']

    def votingProcess(self, squareWave):
        peaksVotersAndInfo = {}
        dipsVotersAndInfo = {}
        peaksVoterInfo = []
        for i in reversed(self.stepsOfPercentError):
            peaksVoterInfo.append(i)
        dipsVoterInfo =  self.stepsOfPercentError
        peaksVotersAndInfo = self.calculateVoterWeight(peaksVoterInfo, 'p')
        dipsVotersAndInfo = self.calculateVoterWeight(dipsVoterInfo, 'd')
        return self.finalizeVictor(squareWave, peaksVotersAndInfo, dipsVotersAndInfo)

    def calculateVoterWeight(self, percentErrorInfo, peaksOrDips):
        allVotesForStructures = {}
        for i in percentErrorInfo:
            for x in self.voterInfo[i]:
                if(x[0] == peaksOrDips):
                    keyString = str(x[1]) + "-" + str(x[2])
                    if(not(keyString in allVotesForStructures)):
                        allVotesForStructures.setdefault(keyString, [0])
                        allVotesForStructures = self.checkForSubStructures(allVotesForStructures, x[1], x[2], keyString)
                        allVotesForStructures = self.addUpVotes(allVotesForStructures, keyString)
                    else:
                        allVotesForStructures = self.addUpVotes(allVotesForStructures, keyString)
        return allVotesForStructures

    def checkForSubStructures(self, allVotesForStructures, currentWidthStart, currentWidthEnd, keyString):
        for i in allVotesForStructures:
            if(i != keyString):
                previousStructure = i.split("-")
                previousStructureStart = int(previousStructure[0])
                previousStructureEnd = int(previousStructure[1])
                if((previousStructureStart >= currentWidthStart) and (previousStructureEnd <= currentWidthEnd)):
                    allVotesForStructures[keyString].append(i)
        return allVotesForStructures

    def addUpVotes(self, allVotesForStructures, keyString):
        allVotesForStructures[keyString][0] = allVotesForStructures[keyString][0] + 1
        for i in range(1,len(allVotesForStructures[keyString])):
            allVotesForStructures[allVotesForStructures[keyString][i]][0] =  allVotesForStructures[allVotesForStructures[keyString][i]][0] + 1
        return allVotesForStructures

    def finalizeVictor(self, squareWave, peaksVotersAndInfo, dipsVotersAndInfo):
        for i in range(0,len(squareWave)):
            totalPeakVotes = 0
            totalDipVotes = 0
            for x in peaksVotersAndInfo:
                temp = x.split("-")
                strat = int(temp[0])
                end = int(temp[1])
                if((strat <= i) and (end >= i)):
                    totalPeakVotes = totalPeakVotes + peaksVotersAndInfo[x][0]
            for y in dipsVotersAndInfo:
                temp = y.split("-")
                strat = int(temp[0])
                end = int(temp[1])
                if((strat <= i) and (end >= i)):
                    totalDipVotes = totalDipVotes + dipsVotersAndInfo[y][0]
            if(totalPeakVotes > totalDipVotes):
                squareWave[i] = 1
            elif(totalDipVotes > totalPeakVotes):
                squareWave[i] = 0
            else:
                squareWave[i] = 1
        return squareWave

    def finalCheckOnKmeans(self, squareWave, data):
        stretchingData = []
        nonStretchDataInfo = []
        for i in range(0,len(data)):
            if(squareWave[i] == 1):
                stretchingData.append(data[i])
            else:
                temp = []
                temp.append(data[i])
                temp.append(i)
                nonStretchDataInfo.append(temp)
        minValueStretch = min(stretchingData)
        for x in nonStretchDataInfo:
            if(x[0] >= minValueStretch):
                squareWave[x[1]] = 1
        return squareWave

    def LastCheck(self, squareWave):
        newSquareWave = []
        stillGatheringInfo = True
        noiseWidth = float(self.noisePeaks)/self.deltat
        squareWaveInfo = []
        countPeaks = 0
        countDips = 0
        stretchingWidths, nonStretchingWidths = self.findWidth(squareWave)
        while(stillGatheringInfo):
            if(squareWave[0] == 1):
                if(countPeaks <= (len(stretchingWidths)-1)):
                    peakInfo = ['p']
                    peakInfo.append(stretchingWidths[countPeaks])
                    countPeaks = countPeaks + 1
                    squareWaveInfo.append(peakInfo)
                if(countDips <= (len(nonStretchingWidths)-1)):
                    dipInfo = ['d']
                    dipInfo.append(nonStretchingWidths[countDips])
                    countDips = countDips + 1
                    squareWaveInfo.append(dipInfo)
            else:
                if(countDips <= (len(nonStretchingWidths)-1)):
                    dipInfo = ['d']
                    dipInfo.append(nonStretchingWidths[countDips])
                    countDips = countDips + 1
                    squareWaveInfo.append(dipInfo)
                if(countPeaks <= (len(stretchingWidths)-1)):
                    peakInfo = ['p']
                    peakInfo.append(stretchingWidths[countPeaks])
                    countPeaks = countPeaks + 1
                    squareWaveInfo.append(peakInfo)
            if((countDips + countPeaks) == (len(stretchingWidths) + len(nonStretchingWidths))):
                stillGatheringInfo = False
        waitingToBeTested = []
        validState = []
        for i in range(0,len(squareWaveInfo)):
            if((squareWaveInfo[i][1] <= noiseWidth)):
                waitingToBeTested.append(i)
            elif(squareWaveInfo[i][1] >= noiseWidth):
                validState.append(i)
            if(len(validState) == 2):
                if((validState[0] + 1) == validState[1]):
                    validState = [validState[1]]
                else:
                    totalPeakWidths = 0
                    numberOfPeaks = 0
                    totalDipWidths = 0
                    numberOfDips = 0
                    meanPeak = 0
                    meanDip = 0
                    for x in waitingToBeTested:
                        if(squareWaveInfo[x][0] == 'p'):
                            totalPeakWidths = squareWaveInfo[x][1] + totalPeakWidths
                            numberOfPeaks = numberOfPeaks + 1
                        else:
                            totalDipWidths = squareWaveInfo[x][1] + totalDipWidths
                            numberOfDips = numberOfDips + 1
                    if(numberOfPeaks > 0):
                        meanPeak = float(totalPeakWidths)/numberOfPeaks
                    if(numberOfDips > 0):
                        meanDip = float(totalDipWidths)/numberOfDips
                    if(len(waitingToBeTested) == 1):
                        if((squareWaveInfo[validState[0]][0] == 'p') and (squareWaveInfo[validState[1]][0] == 'p')):
                            for y in waitingToBeTested:
                                squareWaveInfo[y][0] =  'p'
                        else:
                            for y in waitingToBeTested:
                                squareWaveInfo[y][0] =  'd'
                    elif(meanPeak > meanDip):
                        for y in waitingToBeTested:
                            squareWaveInfo[y][0] =  'p'
                    else:
                        for y in waitingToBeTested:
                            squareWaveInfo[y][0] =  'd'
                    validState = [validState[1]]
                    waitingToBeTested = [] 
        for t in range(validState[0] + 1,len(squareWaveInfo)):
            if(squareWaveInfo[t][1] <= noiseWidth):
                squareWaveInfo[t][0] =  '-'
        for b in squareWaveInfo:
            for v in range(0,b[1]):
                if(b[0] == 'p'):
                    newSquareWave.append(1)
                elif(b[0] == 'd'):
                    newSquareWave.append(0)
                else:
                    newSquareWave.append(0)
        return newSquareWave

    def getPredictionData(self, squareWave, relative_timestamp):
        startTimesForPeaks = []
        for i in range(len(squareWave)):
            if(squareWave[i] == 1 and ((i != 0) and (squareWave[i - 1] == 0))):
                startTimesForPeaks.append(relative_timestamp[i])
        differencesBetweenPeakToPeak = list(np.diff(startTimesForPeaks))
        if(len(self.peakDifference) == 0):
            self.peakDifference = differencesBetweenPeakToPeak
            self.peakStartConstant = startTimesForPeaks
            self.size = len(self.peakStartConstant)
        else:
            if((startTimesForPeaks[-1] - self.peakStartConstant[-1]) > 1): 
                self.peakDifference.append(differencesBetweenPeakToPeak[-1])
                self.peakStartConstant.append(startTimesForPeaks[-1])
        
    def makePrediction(self, squareWave):
        count = 0
        trainningData = []
        for i in reversed(self.peakStartConstant):
            count = count + 1
            trainningData.append(i)
            if(count == 4):
                break
        temp = []
        for x in reversed(trainningData):
            temp.append(x)
        if(self.size == 0):
            self.size = len(self.peakStartConstant)
        else:
            if(self.size < len(self.peakStartConstant)):
                temp.pop() #Remove the value we are trying to predict
                temp = np.asarray(temp, dtype='float64')
                APS = (self.predictModel_AP_Single.predict(temp[-1])[0]) - self.peakStartConstant[-1]
                GNGDS = (self.predictModel_GNGD_Single.predict(temp[-1])[0]) - self.peakStartConstant[-1]
                NLMS = (self.predictModel_NLMS_Single.predict(temp[-1])[0]) - self.peakStartConstant[-1]
                APM = self.predictModel_AP_Multiple.predict(temp) - self.peakStartConstant[-1]
                GNGDM = self.predictModel_GNGD_Multiple.predict(temp) - self.peakStartConstant[-1]
                NLMM = self.predictModel_NLMS_Multiple.predict(temp) - self.peakStartConstant[-1]
                print "Actual Start Time: ", self.peakStartConstant[-1]
                print "Training Data For Single: ", temp[-1]
                print "Training Data For Multiple: ", temp
                print "--------------------------------------------------------------------"
                print "Difference of Prediction and Actual (AP Single): ", APS
                print "Difference of Prediction and Actual (GNGD Single): ", GNGDS
                print "Difference of Prediction and Actual (NLMS Single): ", NLMS
                print "Difference of Prediction and Actual (AP Multiple): ", APM
                print "Difference of Prediction and Actual (GNGD Multiple): ", GNGDM
                print "Difference of Prediction and Actual (NLMS Multiple): ", NLMM
                print "Mean of the Differences: ", float((APS + GNGDS + NLMS + APM + GNGDM + NLMM))/6, "\n"
                self.size = len(self.peakStartConstant)
                singleData = np.asarray(temp[-1], dtype='float64')
                self.predictModel_AP_Single.adapt(float(self.peakStartConstant[-1]), singleData)
                self.predictModel_GNGD_Single.adapt(float(self.peakStartConstant[-1]), singleData)
                self.predictModel_NLMS_Single.adapt(float(self.peakStartConstant[-1]), singleData)
                self.predictModel_AP_Multiple.adapt(float(self.peakStartConstant[-1]), temp)
                self.predictModel_GNGD_Multiple.adapt(float(self.peakStartConstant[-1]), temp)
                self.predictModel_NLMS_Multiple.adapt(float(self.peakStartConstant[-1]), temp)
            
            
    # IBI paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4278369/
    def armu(self, periods, inc):
        # http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016#AR
    	mdl = self.arfit(periods, inc) 
    
    	# Update weights
    	if len(mdl) > 2:
    		mn = mdl[0]
    		for j in range(1, len(mdl)):
    			mn = mn + (mdl[j] * periods[j])
    		return mn
    	else:
    		return 0
        
    def ibipdf(self, t, currentabsolutepeaktime, var, periods, inc, window_size=6):
    	# Check edge cases
    	if t - currentabsolutepeaktime <= 0:
    		diff1 = 1
    		diff2 = 0
    	else:
    		diff1 = ((t-currentabsolutepeaktime)**2)
    		diff2 = math.log(t - currentabsolutepeaktime)
    	if len(periods) == 0:
    		return 0
    	if math.sqrt(2 * math.pi * var * diff1) == 0:
    		return 0
    	if var == 0:
    		return 0
    	armu_val = self.armu(periods[:window_size],inc)
    	if armu_val == 0:
    		return 0
        
            
    def filt(self, observedhist, estimatedhist):   
        taps = 2
    	step = 1
    	projorder = 1
    	y,e,w = adaptfilt.ap(observedhist, estimatedhist, taps, step, projorder)
    	return y,e,w   

    def generator(self, mean, std, bvec, size=50):   
        result = []
        val = np.random.normal(mean, std, size=1)[0]
    	
    	for i in range(size):
    		if len(result) >= len(bvec):
    			val = 0
    			for j in range(len(bvec)):
    				val = val + (bvec[j] * result[-1*(j+1)])
    		else:
    			val = np.random.normal(mean, std, size=1)[0]
    			
    		result.append(val)
    	return result

    def arfit(self, inputs, inc, order=10):
	# Check edge cases
    	start = 2
    	if len(inputs) < 3:
    		return [0,0]
    	if len(inputs) < order:
    		order = len(inputs)-1
       
    	# Initialize empty matrices
    	A = np.zeros((order-start,order-start))
    	b = np.zeros(order-start)
       
       	# Compute fitted weights
    	for i in range(start, order):
    		for j in range(len(inputs[:i])-1):
    			A[i-start][j] = inputs[j]
    		b[i-start] = inputs[i]    
    
    	# Solve system of equations
    	A = np.array(A)
    	b = np.array(b)
    	weights = np.linalg.solve(A, b)
    	return weights

    def mle_ibi(self, H, t, inc, a=0.3, t_horizon=100):
    	n = len(H) - 1 # Number of peak observations (first element is absolute observation -> count-1)
    	w = lambda w_a, w_t, w_u: math.exp(-w_a*(w_t-w_u)) # Weight function (leaky integrator)
    	u_n = H[n]
    
    	# Compute MLE summation
    	total_sum = 0.0
    	for i in range(2, n):
    		curr_H = [sum(H[-i:])] + H[-i:]
    		abs_obs = curr_H[0] # Absolute observation
    		w_k = curr_H[1]
    		ibi_val = self.ibipdf(t, abs_obs, np.var(curr_H[1:]), curr_H[1:],inc)
    		sum1 = w(a, t, abs_obs) * ibi_val
    		total_sum += sum1
    	
    	# Check edge case for integral
    	end_int = H[0]
    	if end_int < t:
    		end_int = t_horizon+t
    	elif end_int > t_horizon+t:
    		end_int = t_horizon+t
    
    	# Compute actual integral and sum to rest of summation
    	integral_res = self.integrate.quad( lambda x: self.ibipdf(x, H[0], np.var(H[1:]), H[1:],inc), t-H[0], end_int)
    	sum2 = w(a, t, u_n) * integral_res[0]
    	total_sum += sum2
    	return total_sum
    
    def generate_future_H(self, w_next, curr_H, curr_t):
    	res = [w_next+curr_t] + [w_next + curr_t - sum(curr_H[1:])] + curr_H[1:]
    	return res
    
    def print_and_plot(self, peaks):
    
        observed = peaks
        periods = np.diff(peaks)
    
        estimated = []
        este = []
        ye = []
        maxxs = []
    
    	# Solve for H that gives maximal mle_val 
        for i in range(3, len(observed)):
    		tmp1 = [observed[i-1]] 
    		tmp2 = np.diff(observed[:i-1]).tolist()
    		H = tmp1 + tmp2
    		t = observed[i]
    
    		# Choose max index from generated mle_ibi
    		poss_vals = np.linspace(0, 10, 1000)
    		ibi_vals = [self.mle_ibi(self.generate_future_H(x, H, t), t, i) for x in poss_vals]
    		max_indx = np.argmax(ibi_vals)
    		max_x = poss_vals[max_indx]
    
    		# Print estimate of next observation
    		if max_x > 0:
    			print("Current observation is", observed[i-1], "I predict the next peak will be in time", max_x, "at", observed[i-1]+max_x)
    			estimated.append(observed[i-1]+max_x)
    			maxxs.append(max_x)
    
    	# Filter observations and estimates
        for i in range(len(estimated)-1):
    		ee = estimated[i] - observed[i+2]
    		este.append(ee)
        y,e,w = self.filt(estimated, observed)
        for i in range(2, len(y)):
    		ee = y[i] - observed[i+1]
    		ye.append(ee)
    
    	# Print estimates
        print('estimated', estimated)
        print('observed', observed)
        print('y', y)
        print('e', e)
        print('este', este)
        print('ye', ye)
        print('periods', periods)
        print('maxxs', maxxs)
    
    	# Format estimates
        observeddiff = periods
        ydiff = np.diff(y)
        estimateddiff = maxxs
    
    	# Compute RMS goodness of fit of y to observed for post-filtering, estimated to observed for pre-filtering predicted estimate fit
        burnin=0
        msepre = mean_squared_error(observeddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:], estimateddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:])
        msepost = mean_squared_error(observeddiff[-1*min(len(observeddiff), len(ydiff))+burnin:], ydiff[-1*min(len(observeddiff), len(ydiff))+burnin:])
        print("RMS error on predictions pre and post adaptive filtering", math.sqrt(msepre), math.sqrt(msepost))
    
    	# Compute chi square goodness of fit
        chipre = scipy.stats.chisquare(estimateddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:], f_exp=observeddiff[-1*min(len(observeddiff), len(estimateddiff))+burnin:])
        chipost = scipy.stats.chisquare(ydiff[-1*min(len(observeddiff), len(ydiff))+burnin:], f_exp=observeddiff[-1*min(len(observeddiff), len(ydiff))+burnin:])
        print("Chi Squared goodness of fit pre and post adaptive filtering", chipre, chipost)
    
    	# Plot figures
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Time (sec)")
        ax1.plot(range(len(observeddiff)), observeddiff, 'r-', label="Observed")
        ax1.plot(range(1,len(estimateddiff)+1), estimateddiff, 'go', label="Estimated a priori")
        ax1.plot(range(len(estimated)-len(este)+1, len(ydiff)+len(estimated)-len(este)+1), ydiff, 'b-', label="Corrected by Adaptive Filter")
        ax1.legend(loc=0)
        ax2.plot(range(len(estimated)-len(este), len(estimated)-1), este[1:], 'm-', label="Model Estimate Error")
        ax2.plot(range(len(estimated)-len(este)+1, len(e)+len(estimated)-len(este)+1), e, 'c-', label="Adaptive Error")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Time (sec)")
        ax2.legend(loc=0)
        plt.suptitle("Model Predicted Interbreath Interval (IBI) Times and Adaptive Filter from Observed Error Terms")
        plt.show()

                 
    def process(self):
        super(VentilatorMeasure , self).process()
        self.newmeasureavailable = False
        data = []
        relative_timestamp = []
        starttime=(self.sensor.iteration * self.sensor.iterationsize - self.gobackn) * self.sensor.timescale
        if starttime < 0: # ignore incomplete data sets pending the gobackn window
            return
        nontrivialmagnitude = self.isnontrivialmagnitude(self.sensor, trainperiod = 900) # for t-test to ensure mean power magnitude is nontrivial
        if nontrivialmagnitude == True:   
        
            #Garbing the data from the server ever 0.5 seconds
            rows = self.sensor.df.query('relative_timestamp >= ' + str((self.sensor.iteration * self.sensor.iterationsize - self.gobackn) * self.sensor.timescale) + ' and relative_timestamp < ' + str(self.sensor.iteration * self.sensor.timescale * self.sensor.iterationsize))
            if self.deltat > 0:
                rows = self.sensor.constantdeltat(rows, deltat=str(int(self.deltat * self.sensor.timescale)) + 'U')
            for i, row in rows.iterrows(): 
                data.append(float(row['prx_moving_parts_deoscillated'])) #prx_moving_parts_deoscillated
                relative_timestamp.append(float(row['relative_timestamp']))
            data = self.sensor.interpnan(data)
            relative_timestamp = self.sensor.interpnan(relative_timestamp)    
            if len(data) == 0 or len(relative_timestamp) == 0:
                return # don't return a rate if there is no data
            data = np.asarray(data, dtype='float64')
            relative_timestamp = np.asarray(relative_timestamp, dtype='float64')
            
            #Convert microseconds to seconds
            temp = []
            for i in relative_timestamp:
                temp.append(float(i)/1000000)
            relative_timestamp = temp
            
            data = data * -1 # Flip the data so it appears in the correct orientation when graphed 
            
            #Classification of respiratory activity and non-respiratory activity and prediction
            squareWave = self.noiseReduction(data, relative_timestamp)
            
            peaks = []
            
            self.print_and_plot(peaks)
            


        else:
            return 
