from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
import numpy as np
from PIL import Image
import pylandstats as pls
import os


def figureOfMerit_TJ(y_true, y_pred, x_true):
    hit = miss = false = 0
    number1 = number2 = number3 = 0
    for i in range(y_true.shape[0]):
        if x_true[i] == 0 and y_true[i] == 2 and y_pred[i] == 2:
            hit += 1
        if x_true[i] == 0 and y_true[i] == 0 and y_pred[i] == 2:
            false += 1
        if x_true[i] == 0 and y_pred[i] == 0 and y_true[i] == 2:
            miss += 1
        if x_true[i] == 0 and y_pred[i] == 2:
            number1 += 1
        if x_true[i] == 0 and y_true[i] == 2:
            number2 += 1
        if x_true[i] == 0 and y_true[i] == 1 and y_pred[i] == 2:
            number3 += 1

    fom = hit / (hit + miss + false)
    print(np.where(x_true == 0)[0].shape, x_true.shape)
    print(hit + miss, number2, hit + false, number1)
    print("fom", fom, "hit", hit, "miss", miss, "false", false)
    return fom


def accuracy(newLandPre, newLandTrue):
    oa = accuracy_score(newLandTrue, newLandPre)
    kappa_Value = cohen_kappa_score(newLandTrue, newLandPre)
    print("oa值为 %f" % oa, "\tkappa值为 %f" % kappa_Value)


def accuracy2(newLandPre, newLandTrue, oldLandTrue):
    fom = figureOfMerit_TJ(newLandTrue, newLandPre, oldLandTrue)
    print("fom值为 %f" % fom)


if __name__ == '__main__':
    ROOT_DIR = "./DATA/Wuhan_F/LU/"
    DATA_DIR = "./DATA/"
    NEIGHBOUR = 9
    OLD_YEAR = str(2000)
    NEW_YEAR = str(2010)
    OUTPUT_DIR1 = os.path.join("./DATA/CA_DATA/", OLD_YEAR)
    OUTPUT_DIR2 = os.path.join("./DATA/CA_DATA/", NEW_YEAR)

    dataLandYear1 = np.array(Image.open(ROOT_DIR + OLD_YEAR + ".png"))
    dataLandYear2 = np.array(Image.open(ROOT_DIR + NEW_YEAR + ".png"))
    dataLandYear1 = np.array(dataLandYear1 / 125, dtype=np.uint8)
    dataLandYear2 = np.array(dataLandYear2 / 125, dtype=np.uint8)

    nonUrbanIndex1 = np.load(os.path.join(OUTPUT_DIR1, str(NEIGHBOUR), 'nonUrbanIndex.npy'))
    # nonUrbanIndex2 = np.load(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), 'nonUrbanIndex.npy'))
    inIndex1 = np.load(os.path.join(OUTPUT_DIR1, str(NEIGHBOUR), 'inIndex.npy'))
    # inIndex2 = np.load(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), 'inIndex.npy'))
    simulateLand1 = np.load(os.path.join(OUTPUT_DIR1, str(NEIGHBOUR), 'simulateArray.npy')) - 1
    # simulateLand2 = np.load(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), 'simulateArray.npy')) - 1

    land2 = dataLandYear2.ravel()[inIndex1]
    land1 = dataLandYear1.ravel()[inIndex1]
    land2_simulate = simulateLand1.ravel()[inIndex1]

    land2_2 = dataLandYear2.ravel()[nonUrbanIndex1]
    land1_2 = dataLandYear1.ravel()[nonUrbanIndex1]
    land2_simulate_2 = simulateLand1.ravel()[nonUrbanIndex1]

    print("Calibration:")
    print("将old数据作为预测:")
    accuracy(land1, land2)
    accuracy2(land1_2, land2_2, land1_2)
    print("实际预测:")
    accuracy(land2_simulate, land2)
    accuracy2(land2_simulate_2, land2_2, land1_2)