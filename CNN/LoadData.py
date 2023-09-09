# -*- coding: UTF-8 -*-
import numpy as np
from osgeo import gdal
import os
from PIL import Image
import random
import pandas as pd
import math

np.random.seed(6)

# 载入tif文件
def loadGridData(name):
    dataset = gdal.Open(name)
    im_height = dataset.RasterYSize
    im_width = dataset.RasterXSize
    data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return data


# 载入某一年的临近因子
def loadProximityDataEachYear(year, dir):
    dataProximity = []
    for root, dirs, files in os.walk(dir + 'Wuhan_F/DIS'):
        for name in sorted(files):
            if str(year) in name:
                file = os.path.join(root, name)
                # if "railways" in name or "secondary" in name or "tertiary" in name or "primary" in name \
                #         or "trunk" in name or "motorway" in name:
                print(name)
                data = loadGridData(file)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataProximity.append(data)

    factors = dataProximity[0]
    for i in range(1, len(dataProximity)):
        factors = np.concatenate((factors, dataProximity[i]), axis=-1)
    return factors


def loadPopDataEachYear(year, dir):
    dataPop = None
    for root, dirs, files in os.walk(dir + 'Wuhan_F/POP'):
        for name in sorted(files):
            if str(year) in name:
                print(name)
                file = os.path.join(root, name)
                data = loadGridData(file)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                dataPop = data
    return dataPop


# 载入自然因子, 如DEM, slope
def loadNatureData(dir):
    dataNature = []
    for root, dirs, files in os.walk(dir + 'Wuhan_F/NA'):
        for name in sorted(files):
            if 'bianJie' not in name:
                print(name)
                file = os.path.join(root, name)
                data = loadGridData(file)
                data = data.reshape((data.shape[0], data.shape[1], 1))
                # data = data.ravel().reshape((-1, 1))
                dataNature.append(data)

    factors = dataNature[0]
    for i in range(1, len(dataNature)):
        factors = np.concatenate((factors, dataNature[i]), axis=-1)
    return factors


def generatePositionData(row, col):
    result = np.zeros((row, col, 2))
    for i in range(row):
        for j in range(col):
            result[i, j, 0] = i
            result[i, j, 1] = j
    # print(np.min(result.reshape(row * col, -1)[:, 0]), np.max(result.reshape(row * col, -1)[:, 0]))
    # print(row, col)
    return result


# 提取城市边界内部的栅格
def extractCityFeature(cityArray, landArray, RADIUS):
    row = cityArray.shape[0]
    col = cityArray.shape[1]
    noWaterArray = np.zeros((row, col), dtype=np.uint32)
    waterArray = np.zeros((row, col), dtype=np.uint32)
    noUrbanArray = np.zeros((row, col), dtype=np.uint32)
    inArray = np.zeros((row, col), dtype=np.uint32)
    for i in range(RADIUS, row - RADIUS):
        for j in range(RADIUS, col - RADIUS):
            if cityArray[i][j] != 15:
                inArray[i][j] = 255
                if landArray[i][j] != 1:
                    noWaterArray[i][j] = 255
                if landArray[i][j] == 1:
                    waterArray[i][j] = 255
                if landArray[i][j] == 0:
                    noUrbanArray[i][j] = 255
    print("城市(2)",
          np.where(noWaterArray.ravel() == 255)[0].shape[0] - np.where(noUrbanArray.ravel() == 255)[0].shape[0],
          "非城市用地(0)", np.where(noUrbanArray.ravel() == 255)[0].shape[0],
          "水(1)", np.where(waterArray.ravel() == 255)[0].shape[0])
    return np.where(noUrbanArray.ravel() == 255)[0], noWaterArray, waterArray, noUrbanArray, \
           np.where(inArray.ravel() == 255)[0]


def saveNPYEveryOne(array, label, dir, year):
    ans = []
    for i in range(array.shape[0]):
        if i % 1000 == 0:
            print(i)
        np.save(dir + year + "/" + str(i), array[i])
        ans.append([dir + year + "/" + str(i) + ".npy", label[i][0]])
    test = pd.DataFrame(data=ans)
    test.to_csv(dir + year + 'test.csv', encoding='gbk', index=False, header=False)


def saveCSV_trainAndval(label, positiveList, negativeList, dir, year):
    trainNum = int(positiveList.shape[0] * 0.99)
    trainList = []
    valList = []
    for i in range(trainNum):
        trainList.append([dir + year + "/" + str(positiveList[i]) + ".npy", label[positiveList[i]][0]])
        trainList.append([dir + year + "/" + str(negativeList[i]) + ".npy", label[negativeList[i]][0]])

    for i in range(trainNum, positiveList.shape[0]):
        valList.append([dir + year + "/" + str(positiveList[i]) + ".npy", label[positiveList[i]][0]])
        valList.append([dir + year + "/" + str(negativeList[i]) + ".npy", label[negativeList[i]][0]])

    train = pd.DataFrame(data=trainList)
    train.to_csv(dir + year + 'train.csv', encoding='gbk', index=False, header=False)

    test = pd.DataFrame(data=valList)
    test.to_csv(dir + year + 'val.csv', encoding='gbk', index=False, header=False)


def saveCSV_trainAndval_prevalences(label, positiveList, negativeList, dir, year, neighbour):
    trainNum1 = int(positiveList.shape[0] * 0.90)
    trainNum2 = int(negativeList.shape[0] * 0.90)

    trainList = []
    valList = []
    for i in range(trainNum1):
        trainList.append(
            [dir + year + "_" + str(neighbour) + "/" + str(positiveList[i]) + ".npy", label[positiveList[i]][0]])

    for i in range(trainNum2):
        trainList.append(
            [dir + year + "_" + str(neighbour) + "/" + str(negativeList[i]) + ".npy", label[negativeList[i]][0]])

    for i in range(trainNum1, positiveList.shape[0]):
        valList.append(
            [dir + year + "_" + str(neighbour) + "/" + str(positiveList[i]) + ".npy", label[positiveList[i]][0]])

    for i in range(trainNum2, negativeList.shape[0]):
        valList.append(
            [dir + year + "_" + str(neighbour) + "/" + str(negativeList[i]) + ".npy", label[negativeList[i]][0]])

    train = pd.DataFrame(data=trainList)
    train.to_csv(dir + year + "_" + str(neighbour) + "_train.csv", encoding='gbk', index=False, header=False)

    test = pd.DataFrame(data=valList)
    test.to_csv(dir + year + "_" + str(neighbour) + "_val.csv", encoding='gbk', index=False, header=False)


# 制作X:提取邻域的数据
def extractNeighbourFeature(array, indexArray, neighbour, space):
    height = array.shape[0]
    width = array.shape[1]
    number = height * width
    data = np.zeros((number, neighbour, neighbour, 12), dtype=np.float16)
    data[:, :, :, -1] = -1
    id = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if indexArray[i][j] == 0:
                id = id + 1
            else:
                data[id, :, :, :] = array[i - space: i + space + 1, j - space: j + space + 1, :]
                id = id + 1

    # 最后一维有值不可能为-1
    n = 0
    for i in range(data.shape[0]):
        if data[i, 0, 0, -1] != -1:
            n += 1
    print("提取每一个有效栅格的邻域，经检验，有效数量为", n, "输出结果形状为", data.shape)
    return data


def extractNeighbourFeature_2(array, indexArray, neighbour, space):
    number = np.where(indexArray.ravel() == 255)[0].shape[0]
    data = np.zeros((number, neighbour, neighbour, 12), dtype=np.float16)
    data[:, :, :, -1] = -1
    id = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if indexArray[i][j] != 0:
                data[id, :, :, :] = array[i - space: i + space + 1, j - space: j + space + 1, :]
                id = id + 1

    # 最后一维有值不可能为-1
    n = 0
    for i in range(data.shape[0]):
        if data[i, 0, 0, -1] != -1:
            n += 1
    print("提取每一个有效栅格的邻域，经检验，有效数量为", n, "输出结果形状为", data.shape)
    return data


# 制作X:提取邻域的数据
def extractNeighbourFeature2(array, landData, indexArray, neighbour, space):
    height = array.shape[0]
    width = array.shape[1]
    number = height * width
    data = np.zeros((number, neighbour, neighbour, 13), dtype=np.float16)
    data[:, :, :, -1] = -1
    id = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if indexArray[i][j] == 0:
                id = id + 1
            else:
                data[id, :, :, :-1] = array[i - space: i + space + 1, j - space: j + space + 1, :]
                data[id, :, :, -1] = landData[i - space: i + space + 1, j - space: j + space + 1]
                id = id + 1

    # 最后一维有值不可能为-1
    n = 0
    for i in range(data.shape[0]):
        if data[i, 0, 0, -1] != -1:
            n += 1
    print("提取每一个有效栅格的邻域，经检验，有效数量为", n, "输出结果形状为", data.shape)
    return data


def extractNeighbourFeature2_2(array, landData, indexArray, neighbour, space, channel):
    number = np.where(indexArray.ravel() == 255)[0].shape[0]
    data = np.zeros((number, neighbour, neighbour, channel + 1), dtype=np.float16)
    data[:, :, :, -1] = -1
    id = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if indexArray[i][j] != 0:
                data[id, :, :, :-1] = array[i - space: i + space + 1, j - space: j + space + 1, :]
                data[id, :, :, -1] = landData[i - space: i + space + 1, j - space: j + space + 1] / 2
                id = id + 1

    # 最后一维有值不可能为-1
    n = 0
    for i in range(data.shape[0]):
        if data[i, 0, 0, -1] != -1:
            n += 1
        if data[i, 0, 0, -1] >= 2:
            print('yyy')

    print("提取每一个有效栅格的邻域，经检验，有效数量为", n, "输出结果形状为", data.shape)
    return data


def extractNeighbourFeature2_3(array, landData, indexArray, neighbour, space, channel, label, dir, year, flag=True):
    ans = []
    id = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if indexArray[i][j] != 0:
                if id % 1000 == 0:
                    print(id)
                data = np.zeros((neighbour, neighbour, channel + 1), dtype=np.float16)
                data[:, :, :-1] = array[i - space: i + space + 1, j - space: j + space + 1, :]
                data[:, :, -1] = landData[i - space: i + space + 1, j - space: j + space + 1] / 2
                if flag:
                    np.save(dir + year + "_" + str(neighbour) + "/" + str(id), data)
                ans.append([dir + year + "_" + str(neighbour) + "/" + str(id) + ".npy", label[id][0]])
                id = id + 1
    test = pd.DataFrame(data=ans)
    test.to_csv(dir + year + "_" + str(neighbour) + '_test.csv', encoding='gbk', index=False, header=False)


# 制作Y:提取变化的区域
def extractChangeArea(newData, oldData, indexArray):
    row = newData.shape[0]
    col = newData.shape[1]
    new_array = np.zeros((row, col), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            if indexArray[i][j] == 255:
                if oldData[i][j] != 0:
                    print("ERROR")
                if newData[i][j] == 2 and oldData[i][j] == 0:
                    new_array[i, j] = 1
    return new_array


def random_int_list(array):
    # 先判断出正负样本
    indexPositive = np.where(array == 1)[0]
    indexNegative = np.where(array == 0)[0]

    randomListPositive = random.sample(range(0, indexPositive.shape[0]), 10000)
    randomListNegative = random.sample(range(0, indexNegative.shape[0]), 10000)

    randomListPositive = indexPositive[randomListPositive]
    randomListNegative = indexNegative[randomListNegative]

    return randomListPositive, randomListNegative


def random_sample_prevalences(array, rate):
    indexPositive = np.where(array == 1)[0]
    indexNegative = np.where(array == 0)[0]

    positiveNumber = indexPositive.shape[0]
    negativeNumber = indexNegative.shape[0]
    positiveRate = positiveNumber / (positiveNumber + negativeNumber)
    negativeRate = 1 - positiveRate

    totalNumber = array.shape[0]
    totalSampleNumber = int(totalNumber * rate)

    positiveSampleNumber = int(totalSampleNumber * positiveRate)
    negativeSampleNumber = int(totalSampleNumber * negativeRate)

    print("totalNumber", totalNumber, "totalSampleNumber", totalSampleNumber, "positiveSampleNumber",
          positiveSampleNumber, "negativeSampleNumber", negativeSampleNumber)

    randomListPositive = random.sample(range(0, indexPositive.shape[0]), positiveSampleNumber)
    randomListNegative = random.sample(range(0, indexNegative.shape[0]), negativeSampleNumber)

    randomListPositive = indexPositive[randomListPositive]
    randomListNegative = indexNegative[randomListNegative]

    return randomListPositive, randomListNegative


def systematic_sample_prevalences(array, rate):
    indexPositive = np.where(array == 1)[0]
    indexNegative = np.where(array == 0)[0]

    positiveNumber = indexPositive.shape[0]
    negativeNumber = indexNegative.shape[0]
    positiveRate = positiveNumber / (positiveNumber + negativeNumber)
    negativeRate = 1 - positiveRate

    totalNumber = array.shape[0]
    totalSampleNumber = int(totalNumber * rate)

    positiveSampleNumber = int(totalSampleNumber * positiveRate)
    negativeSampleNumber = int(totalSampleNumber * negativeRate)

    print("totalNumber", totalNumber, "totalPositive", positiveNumber, "totalNegative", negativeNumber,
          "totalSampleNumber", totalSampleNumber, "positiveSampleNumber", positiveSampleNumber, "negativeSampleNumber",
          negativeSampleNumber)

    randomListPositive = np.linspace(0, indexPositive.shape[0], num=positiveSampleNumber, endpoint=False, dtype=int)
    randomListNegative = np.linspace(0, indexNegative.shape[0], num=negativeSampleNumber, endpoint=False, dtype=int)

    randomListPositive = indexPositive[randomListPositive]
    randomListNegative = indexNegative[randomListNegative]
    # print(randomListPositive, randomListNegative)
    np.random.shuffle(randomListPositive)
    np.random.shuffle(randomListNegative)

    return randomListPositive, randomListNegative


def systematic_sample_prevalences_2(array, number, rate):
    indexPositive = np.where(array == 1)[0]
    indexNegative = np.where(array == 0)[0]

    positiveNumber = indexPositive.shape[0]
    negativeNumber = indexNegative.shape[0]
    positiveRate = rate
    negativeRate = 1 - positiveRate

    totalNumber = array.shape[0]
    totalSampleNumber = number

    positiveSampleNumber = int(totalSampleNumber * positiveRate)
    negativeSampleNumber = int(totalSampleNumber * negativeRate)

    print("totalNumber", totalNumber, "totalPositive", positiveNumber, "totalNegative", negativeNumber,
          "totalSampleNumber", totalSampleNumber, "positiveSampleNumber", positiveSampleNumber, "negativeSampleNumber",
          negativeSampleNumber)

    randomListPositive = np.linspace(0, indexPositive.shape[0], num=positiveSampleNumber, endpoint=False, dtype=int)
    randomListNegative = np.linspace(0, indexNegative.shape[0], num=negativeSampleNumber, endpoint=False, dtype=int)

    randomListPositive = indexPositive[randomListPositive]
    randomListNegative = indexNegative[randomListNegative]
    print(randomListPositive, randomListNegative)
    np.random.shuffle(randomListPositive)
    np.random.shuffle(randomListNegative)

    return randomListPositive, randomListNegative


def getExpandRate(newData, oldData):
    number1 = number2 = number3 = number4 = number5 = number6 = 0
    for i in range(newData.shape[0]):
        if oldData[i] == 0 and newData[i] == 0:
            number1 += 1
        if oldData[i] == 0 and newData[i] == 1:
            number2 += 1
        if oldData[i] == 0 and newData[i] == 2:
            number3 += 1
        if oldData[i] == 2 and newData[i] == 0:
            number4 += 1
        if oldData[i] == 2 and newData[i] == 1:
            number5 += 1
        if oldData[i] == 2 and newData[i] == 2:
            number6 += 1

    print("getExpandRate:", number1, number2, number3, number4, number5, number6)
    return number3 / (number1 + number2 + number3)


def getStochastic(data, indexArray):
    row = data.shape[0]
    col = data.shape[1]
    stochasticMap = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if indexArray[i][j] == 255:
                stochasticMap[i][j] = 1 + math.pow((-1 * math.log(math.e, random.random())), 0.1)
    return stochasticMap


def getStochastic_2(data, indexArray, radius):
    row = data.shape[0]
    col = data.shape[1]
    stochasticMap = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if indexArray[i][j] == 255:
                array = np.array(data[i - radius: i + radius + 1, j - radius: j + radius + 1])
                number = np.where(array == 2)[0].shape[0]
                space = radius * 2 + 1
                number = number / (space * space - 1)
                if number < 0.1:
                    stochasticMap[i][j] = 1 + math.pow((-1 * math.log(math.e, random.random())), 0.1)
                else:
                    stochasticMap[i][j] = 1
    return stochasticMap


def generateSampleImage(data1, index1, index2, dir):
    newArray = np.zeros((data1.shape[0], data1.shape[1]), dtype="uint8")
    newArray = newArray.reshape((data1.shape[0] * data1.shape[1]))
    for i in range(len(index1)):
        newArray[index2[index1[i]]] = 255
    newArray = newArray.reshape((data1.shape[0], data1.shape[1]))
    img = Image.fromarray(newArray)
    img.save(dir + "positive_distrobution.png")
