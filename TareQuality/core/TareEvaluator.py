import numpy as np
from abc import ABC, abstractmethod
from TareQuality.core.TareDataset import *

class TareValue:
    pass

class TareEvaluator(ABC):
    targetTrace = None
    tareInstances = None
    auxiliaryFeatures = None

    tareValues = None
    tareDifferentials = None

    def __init__(self, dataset:TareDataset, tareInstances:np.array):
        self.targetTrace = dataset.referenceValue
        self.auxiliaryFeatures = dataset.otherValues
        self.tareInstances = tareInstances
        self.tareValues = []
        self.tareDifferentials = []


    def getTareValues(self):
        return self.tareValues

    def getTareDifferentials(self):
        return self.tareDifferentials

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def throwFlag(self):
        pass

    def getModeledValue(self, idx, trailingSamples):
        #capture a linear regression of the trailing n points (excludes the current sample)
        sample = self.targetTrace[self.tareInstances[idx-trailingSamples]:self.targetTrace.tareInstances[idx-1]]
        fit = np.polyfit(np.arange(0, trailingSamples), sample, 1)
        return fit(self.tareInstances[trailingSamples])

    def calculateAllTareValues(self):
        self.tareValues = list(map(lambda x:self.getTareValueAtIdx(x), self.tareInstances))

    def getTareValueAtIdx(self, idx):
        #TODO: Make leading sample window parameter configurable
        leadingSample = self.targetTrace[self.getLeadingSample(idx, 30)]
        return np.median(leadingSample)

    def getLeadingSample(self, idx, sampleLength):
        return range(idx+5, idx+5+sampleLength)

    def calculateTareDifferentials(self):
        assert len(self.tareValues) > 0
        self.tareDifferentials = list(np.diff(np.array(self.tareValues)))


class WobTareEvaluator(TareEvaluator):

    bitDepth = None
    holeDepth = None

    oneStandThreshold = None
    multipleStandThreshold = None
    trailingSampleLength = None

    def __init__(self, dataset:WobTareEvaluatorDataset, tareInstances):
        super().__init__(dataset=dataset,
                         tareInstances=tareInstances)
        self.getDefaultThresholds()

    def getDefaultThresholds(self):
        self.oneStandThreshold = 10
        self.multipleStandThreshold = 10
        self.trailingSampleLength = 20

    def evaluate(self):
        for idx in range(0, len(self.tareInstances)):
            self.tareValues.append(self.getTareValueAtIndex(idx))
            if idx > 0:
                self.tareDifferentials.append(self.getDifferential(idx))

    def throwFlag(self):
        pass

    def getModeledValue(self, idx, trailingSamples):
        pass

    def getTareDifferentials(self):
        pass

    def getTareValueAtIndex(self,idx):
        """
        :param idx:
        :return:
        """
        startTimestamp = self.tareInstances[idx]
        sample = self.targetTrace[startTimestamp:startTimestamp+self.trailingSampleLength]
        return np.average(sample)

    def getDifferential(self, idx):
        """
        :param idx: numeric index representing a timestamp at which a tare is taken
        :return: differential between the tare value observed at this index and that observed at the previous index
        """
        assert idx >= 0
        return self.tareValues[idx] - self.tareValues[idx-1]

    def taredWhileOnBottom(self) -> bool:
        #TODO: return positive if bit depth == hole depth at the index of a tare
        pass


class DiffPTareEvaluator(TareEvaluator):

    bitDepth = None
    holeDepth = None
    flowrate = None

    def __init__(self,
                 dataset,
                 tareInstances,
                 ):
        super().__init__(dataset=dataset,
                         tareInstances=tareInstances
                         )

    def evaluate(self):
        pass

    def throwFlag(self):
        pass


    def setFlowrate(self,flowrate):
        self.flowrate = flowrate
        return self

    def setHoleDepth(self,holeDepth):
        self.holeDepth = holeDepth
        return self

    def setBitDepth(self,bitDepth):
        self.bitDepth = bitDepth
        return self

    def checkForSomething(self):
        pass
