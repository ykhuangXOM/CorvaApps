import numpy as np
from TareQuality.core.TareDataset import *
from abc import ABC, abstractmethod


class TareValidator(ABC):

    tareTimestamps = None
    targetData = None
    referenceValues = None
    auxiliaryFeatures = None

    filtered = None

    def __init__(self, dataset:TareDataset, tareTimestamps:np.array):
        self.tareTimestamps = tareTimestamps
        self.targetData = dataset.derivedValue
        self.referenceValues = dataset.referenceValue
        self.auxiliaryFeatures = dataset.otherValues

    @abstractmethod
    def validate(self) -> np.array:
        """
        Given a set of tare data and timestamps of tares, eliminate any anomalies detected as a result of
        initial overfitting. Return a list of timestamps that have passed the logical tests
        """
        pass

    @abstractmethod
    def _validateSingleIndex(self, idx) -> bool:
        """
        :param idx: time-series index position at which to evaluate whether a rising edge is or is not a tare
        :return: boolean representing whether the behavior observed at an index position is indicative of a tare
        """
        pass

    def _verifyNearbyZero(self, idx) -> bool:
        # TODO: Return false if the target value is not zero near the rising/falling edge
        tareTimestamp = self.tareTimestamps[idx]
        leadingValues = self.targetData[tareTimestamp:tareTimestamp + 10]
        mask = (leadingValues > -2) & (leadingValues < 2)
        return any(mask)


class RuleBasedWobTareValidator(TareValidator):

    def __init__(self, dataset:WobTareValidatorDatasetRuleBased, tareTimestamps:np.array):
        super().__init__(dataset=dataset,tareTimestamps=tareTimestamps)

    @property
    def validate(self):
        mask = [self._validateSingleIndex(idx) for idx in range(0, len(self.tareTimestamps))]
        return self.tareTimestamps[mask]

    def _validateSingleIndex(self, idx) -> np.array:
        #TODO: Main method to append rising/falling edges that pass all tests
        return any([self._verifyNearbyZero(idx)])

    def _verifyNearbyZero(self, idx) -> bool:
        #TODO: Return false if the target value is not zero near the rising/falling edge
        tareTimestamp = self.tareTimestamps[idx]
        leadingValues = self.targetData[tareTimestamp:tareTimestamp + 10]
        mask = (leadingValues > -2) & (leadingValues < 2)
        return any(mask)

class RuleBasedDiffTareValidator(TareValidator):

    def __init__(self,dataset:DiffPTareValidatorDatasetRuleBased,tareTimestamps):
        super().__init__(dataset=dataset,tareTimestamps=tareTimestamps)

    def validate(self) -> np.array:
        pass

    def _validateSingleIndex(self,idx) -> bool:
        pass




