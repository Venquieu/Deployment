import os
import glob

import cv2
import numpy as np
import torch

dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
nHeight = 28
nWidth = 28
np.random.seed(31193)

class MyData(torch.utils.data.Dataset):
    def __init__(self, isTrain=True):
        if isTrain:
            self.data = trainFileList
        else:
            self.data = testFileList

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-7])
        label[index] = 1
        return torch.from_numpy(
            data.reshape(1, nHeight, nWidth).astype(np.float32)
        ), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)