import numpy as np
from scipy.signal import medfilt
from scipy.signal import find_peaks

class SlidingWindow:
    #sliding window of odd array length used to implement step detection algorithm

    width = None
    startPosition = None
    midptPosition = None
    endptPosition = None
    firstHalf = None
    secondHalf = None

    def __init__(self, width):
        assert self._isOdd(width)
        self.width = width
        self.startPosition = 0
        self._updateIndices()

    def _updateIndices(self):
        self.midptPosition = int((self.width+1)/2) + self.startPosition
        self.endptPosition = self.width + self.startPosition

    @staticmethod
    def _isOdd(input:int):
        return input %2 != 0

    def reset(self):
        self.startPosition = 0
        self._updateIndices()

    def setPosition(self, idx):
        self.startPosition = idx
        self._updateIndices()

    def increment(self):
        self.startPosition += 1
        self._updateIndices()

class EdgeDetector(object):

    filterKernelSize = None
    stepThreshold = None
    scanningWindow = None
    inputData = None

    regressionDifference = None
    diffTareTimes = None
    wobTareTimes = None

    def __init__(self, inputData):
        self.initializeDefaultScanningWindow()
        self.setDefaultParameters()
        self.inputData = inputData

    def setDefaultParameters(self):
        self.filterKernelSize = 61
        self.stepThreshold = 100 #set step threshold and other configuration details in the TareDataset object

    def setStepThreshold(self,thresh):
        self.stepThreshold = thresh
        return self

    def setFilterKernelSize(self,kernel):
        self.filterKernelSize = kernel
        return self

    def initializeDefaultScanningWindow(self):
        self.setScanningWindow(101)

    def setScanningWindow(self,width):
        self.scanningWindow = SlidingWindow(width)

    def scanRoutine(self, filter=True):

        data = self.inputData
        if filter is True:
            data = medfilt(data, kernel_size=self.filterKernelSize)

        outputData = self.scan(data)
        peaks, _ = find_peaks(outputData, self.stepThreshold)
        return peaks

    @staticmethod
    def singlePieceModel(data):
        # take simple average over the entire sliding window
        return np.average(data) * np.ones(len(data))

    @staticmethod
    def piecewiseModel(data):
        # take simple average over the halves of the sliding window

        oddLength = len(data) % 2 != 0
        bias = 1 if oddLength else 0  # apply a positional bias of 1 if data is odd in length
        firstHalf = data[0:int(np.floor(len(data) / 2) + bias)]
        secondHalf = data[int(np.ceil(len(data) / 2)):len(data)]
        modeled = np.concatenate([np.ones(len(firstHalf)) * np.average(firstHalf),
                                  np.ones(len(secondHalf)) * np.average(secondHalf)])
        return modeled

    def scan(self, data):
        output = np.zeros(int((self.scanningWindow.width - 1) / 2))  # padding zeroes to cover first half of sliding window
        while self.scanningWindow.endptPosition < len(data):
            masked = data[self.scanningWindow.startPosition:self.scanningWindow.endptPosition]
            piecewise = self.piecewiseModel(masked)
            piecewiseResiduals = self.getSquaredResiduals(masked, piecewise)
            singleRegression = self.singlePieceModel(masked)
            singleRegressionResiduals = self.getSquaredResiduals(masked, singleRegression)
            squareResidDifference = abs(np.sum(singleRegressionResiduals) - np.sum(piecewiseResiduals))
            output = np.append(output, squareResidDifference)
            self.scanningWindow.increment()
        self.scanningWindow.reset()

        return output

    @staticmethod
    def getSquaredResiduals(actual, modeled):
        return (actual - modeled)**2


if __name__ == '__main__':
    import pandas as pd
    from matplotlib import pyplot as plt

    df = pd.read_csv(r'C:\Users\yhuang10\PycharmProjects\CorvaApps\validationData\PLU 18 TWR 126H.csv')
    df = df.iloc[320000:440000].reset_index(drop=True)
    tareHookload = df['Hook Load (klbs)'] + df['Weight on Bit (klbs)']

    detector = EdgeDetector(tareHookload)
    edges = detector.scanRoutine()
    plt.plot(df.index, tareHookload)
    plt.plot(df.index, medfilt(tareHookload, 61))
    plt.legend(['Tare Hookload','Tare Hookload with Median Filter'])
    plt.scatter(edges, medfilt(tareHookload, 61)[edges], color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Hookload (klb)')
    plt.show()
