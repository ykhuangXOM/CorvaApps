import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class TareDataset():
    timestamp = None
    referenceValue = None
    derivedValue = None
    otherValues = None
    mandatoryValues = None

    def __init__(self):
        self.otherValues = {}
        self.mandatoryValues = []

    def validate(self):
        for item in [self.timestamp, self.referenceValue, self.derivedValue]:
            assert item is not None

        for key in self.mandatoryValues:
            assert key in self.otherValues.keys()

    def __str__(self):
        header = 'Auxiliary values: \n'
        features = "\n".join(self.otherValues.keys())
        return header + features

    def __getitem__(self, item):
        return self.otherValues[item]

class WobTareValidatorDatasetRuleBased(TareDataset):
    def __init__(self):
        super().__init__()
        self.mandatoryValues = ['bitDepth', 'holeDepth', 'rpm']

class DiffPTareValidatorDatasetRuleBased(TareDataset):
    def __init__(self):
        super().__init__()
        self.mandatoryValues = ['bitDepth',
                                'holeDepth',
                                'gpm',
                                ]

class WobTareValidatorDatasetML(TareDataset):
    def __init__(self):
        super().__init__()
        self.mandatoryValues = ['bitDepth', 'holeDepth', 'rpm']


class WobTareEvaluatorDataset(TareDataset):
    def __init__(self):
        super().__init__()
        self.mandatoryValues = ['bitDepth', 'holeDepth', 'rpm']


class TareDatasetBuilder():
    dataset = None

    def __init__(self):
        self.dataset = TareDataset()

    def build(self):
        self.dataset.validate()
        return self.dataset

    def reset(self):
        self.__init__()

    def setReferenceValue(self,referenceValue):
        self.dataset.referenceValue = referenceValue
        return self

    def setDerivedValue(self,derivedValue):
        self.dataset.derivedValue = derivedValue
        return self

    def setTimestamp(self,timestamp):
        self.dataset.timestamp = timestamp
        return self

    def setAuxiliaryValue(self, key, arbitraryValue):
        self.dataset.otherValues[key] = arbitraryValue
        return self

#TODO: build child items called WOBTareDataset and DPTareDataset

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\yhuang10\PycharmProjects\CorvaApps\validationData\PLU 18 TWR 155.csv')
    dataset = (TareDatasetBuilder().
               setTimestamp(df.index).
               setReferenceValue(df['standpipe_pressure']).
               setDerivedValue(df['diff_press']).
               setAuxiliaryValue('wob', df['weight_on_bit']).
               setAuxiliaryValue('spm3', df['pump_spm_3']).
               build())
    print(dataset)
    print(dataset['wob'])
