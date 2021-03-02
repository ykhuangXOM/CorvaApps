from TareQuality.core.EdgeDetector import *
from TareQuality.core.TareDataset import *
from TareQuality.core.TareEvaluator import *
from TareQuality.core.TareValidator import *

import pandas as pd
import numpy as np
from scipy.signal import medfilt
from matplotlib import pyplot as plt

import logging
from unittest import TestCase

def import_preprocess_testdata():
    df = (pd.read_csv(r'C:\Users\yhuang10\PycharmProjects\CorvaApps\validationData\PLU 18 TWR 126H.csv')
          .iloc[320000:350000]
          .reset_index(drop=True)
          )
    df['Tare Hookload'] = df['Weight on Bit (klbs)'] + df['Hook Load (klbs)']
    df['Filtered Tare Hookload'] = medfilt(df['Tare Hookload'], 61)
    df['Tare SPP'] = df['Standpipe Pressure (psi)'] - df['Differential Pressure (psi)']
    df['Filtered Tare SPP'] = medfilt(df['Tare SPP'], 61)
    return df

def getWobEdgeDetectionDataset(df:pd.DataFrame):
    return df['Filtered Tare Hookload']

def buildWobTareValidatorDataset(df:pd.DataFrame):
    dataset = (TareDatasetBuilder()
               .setReferenceValue(df["Filtered Tare Hookload"])
               .setDerivedValue(df["Weight on Bit (klbs)"])
               .setTimestamp(df.index)
               .setAuxiliaryValue("bitDepth",df["Bit Depth (feet)"])
               .setAuxiliaryValue("holeDepth",df["Hole Depth (feet)"])
               .build()
               )
    return dataset

def buildWobTareEvaluatorDataset(df:pd.DataFrame):
    dataset = (TareDatasetBuilder()
               .setReferenceValue(df["Filtered Tare Hookload"])
               .setDerivedValue("Weight on Bit (klbs)")
               .setTimestamp(df.index)
               .setAuxiliaryValue("bitDepth", df["Bit Depth (feet)"])
               .setAuxiliaryValue("holeDepth", df["Hole Depth (feet)"])
               .setAuxiliaryValue("blockHeight", df["Block Height (feet)"])
               .build()
               )
    return dataset

class TareDatasetTests(TestCase):

    def setUp(self) -> None:
        df = import_preprocess_testdata()
        self.edgeDetectData = getWobEdgeDetectionDataset(df)
        self.validatorData = buildWobTareValidatorDataset(df)
        self.evaluatorData = buildWobTareEvaluatorDataset(df)

    def test_edgeDetectData(self):
        print(self.edgeDetectData)

    def test_validatorData(self):
        print(self.validatorData)

    def test_evaluatorData(self):
        print(self.evaluatorData)

class EdgeDetectorTests(TestCase):

    def setUp(self) -> None:
        df = import_preprocess_testdata()
        self.edgeDetectData = getWobEdgeDetectionDataset(df)
        self.edgeDetector = EdgeDetector(self.edgeDetectData)

    def testEdgeDetectorLogic(self):
        self.edges = self.edgeDetector.scanRoutine()
        print(self.edges)
        self.plot()

    def plot(self):
        timestamp = range(0, len(self.edgeDetectData))
        plt.plot(timestamp, self.edgeDetectData)
        plt.scatter(self.edges, self.edgeDetectData[self.edges],color="orange")
        plt.savefig('outputfiles/edge detection plot')

class ValidatorTests(TestCase):

    def setUp(self) -> None:
        self.edges = np.array([797, 2717, 3912, 4506, 4923, 4926, 5500, 8333, 9562,
                      9704, 11617, 20476, 22151, 25281, 26211])
        masterDataset = import_preprocess_testdata()
        self.dataset = buildWobTareValidatorDataset(masterDataset)
        self.validator = RuleBasedWobTareValidator(self.dataset, self.edges)

    def testValidationLogic(self):
        self.filteredEdges = self.validator.validate
        print(self.filteredEdges)
        self.plot()

    def plot(self):
        plt.plot(self.dataset.timestamp, self.dataset.referenceValue)
        plt.scatter(self.edges, self.dataset.referenceValue[self.edges], color = "green")
        plt.scatter(self.filteredEdges, self.dataset.referenceValue[self.filteredEdges], color="orange")
        plt.savefig('outputfiles/edge validation plot')

class EvaluatorTests(TestCase):

    def setUp(self) -> None:
        self.edges = np.array([797,3912,4923,4926,5500,9562,9704,11617,20476,22151,25281,26211])
        masterDataset = import_preprocess_testdata()
        self.unfilteredTargetData = masterDataset['Tare Hookload']
        self.dataset = buildWobTareEvaluatorDataset(masterDataset)
        self.evaluator = WobTareEvaluator(self.dataset, self.edges)

    def test_tareValueCalculation(self):
        self.evaluator.calculateAllTareValues()
        print(self.evaluator.getTareValues())
        self.plot_tareValues()

    def plot_tareValues(self):
        #plt.plot(self.dataset.timestamp,self.unfilteredTargetData,color="magenta")
        fig,ax = plt.subplots(2,1, figsize=(10,10),sharex=True)
        ax[0].plot(self.dataset.timestamp,self.dataset.referenceValue,color="blue")
        ax[0].scatter(self.evaluator.tareInstances, self.evaluator.tareValues,color="orange")

        ax[1].plot(self.dataset.timestamp,self.dataset.otherValues['blockHeight'])
        bh = self.dataset.otherValues['blockHeight']
        mask = bh[self.evaluator.tareInstances]
        ax[1].scatter(self.evaluator.tareInstances, self.dataset.otherValues['blockHeight'][self.evaluator.tareInstances], color="orange")
        plt.savefig("outputfiles/tare value plot")



