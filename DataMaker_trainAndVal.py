from CNN.LoadData import *
from sklearn import preprocessing

ROOT_DIR = "./DATA/Wuhan_F/LU/"
DATA_DIR = "./DATA/"
OUTPUT_DIR = os.path.join("./DATA/CA_DATA/")


def systematic_sample_prevalences(array, index, rate):
    array = array.ravel().reshape((-1, 1))[index]
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
    np.random.shuffle(randomListPositive)
    np.random.shuffle(randomListNegative)

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


def systematic_sample_prevalences_2(array, index, number, rate):
    array = array.ravel().reshape((-1, 1))[index]
    indexPositive = np.where(array == 1)[0]
    indexNegative = np.where(array == 0)[0]

    positiveNumber = indexPositive.shape[0]
    negativeNumber = indexNegative.shape[0]
    totalNumber = array.shape[0]

    positiveRate = rate
    negativeRate = 1 - positiveRate
    totalSampleNumber = int(number)
    positiveSampleNumber = int(totalSampleNumber * positiveRate)
    negativeSampleNumber = int(totalSampleNumber * negativeRate)

    print("totalNumber", totalNumber, "totalPositive", positiveNumber, "totalNegative", negativeNumber,
          "totalSampleNumber", totalSampleNumber, "positiveSampleNumber", positiveSampleNumber, "negativeSampleNumber",
          negativeSampleNumber)

    if positiveSampleNumber > positiveNumber:
        print("positive exceed")
        randomListPositive = np.linspace(0, indexPositive.shape[0], num=positiveSampleNumber, endpoint=False, dtype=int)
    else:
        randomListPositive = np.linspace(0, indexPositive.shape[0], num=positiveSampleNumber, endpoint=False, dtype=int)

    if negativeSampleNumber > negativeNumber:
        print("negative exceed")
        randomListNegative = np.linspace(0, indexNegative.shape[0], num=negativeSampleNumber, endpoint=False, dtype=int)
    else:
        randomListNegative = np.linspace(0, indexNegative.shape[0], num=negativeSampleNumber, endpoint=False, dtype=int)

    # if positiveSampleNumber > positiveNumber:
    #     print("positive exceed")
    #     randomListPositive = np.random.choice(positiveNumber, positiveSampleNumber, replace=True)
    # else:
    #     randomListPositive = np.random.choice(positiveNumber, positiveSampleNumber, replace=False)
    #
    # if negativeSampleNumber > negativeNumber:
    #     print("negative exceed")
    #     randomListNegative = np.random.choice(negativeNumber, negativeSampleNumber, replace=True)
    # else:
    #     randomListNegative = np.random.choice(negativeNumber, negativeSampleNumber, replace=False)

    randomListPositive = indexPositive[randomListPositive]
    randomListNegative = indexNegative[randomListNegative]

    np.random.shuffle(randomListPositive)
    np.random.shuffle(randomListNegative)

    return randomListPositive, randomListNegative


def saveCSV_trainAndval_prevalences(label, NEIGHBOUR, positiveList, negativeList):
    trainNum1 = int(positiveList.shape[0] * 0.90)
    trainNum2 = int(negativeList.shape[0] * 0.90)
    dir = os.path.join(OUTPUT_DIR, "2000", str(NEIGHBOUR))
    trainList = []
    valList = []
    for i in range(trainNum1):
        if label[positiveList[i]][0] != 1:
            print("YYY")
        trainList.append(
            [os.path.join(dir, "Array", str(positiveList[i])) + ".npy", label[positiveList[i]][0]])

    for i in range(trainNum2):
        if label[negativeList[i]][0] != 0:
            print("YYY")
        trainList.append(
            [os.path.join(dir, "Array", str(negativeList[i])) + ".npy", label[negativeList[i]][0]])

    for i in range(trainNum1, positiveList.shape[0]):
        if label[positiveList[i]][0] != 1:
            print("YYY")
        valList.append(
            [os.path.join(dir, "Array", str(positiveList[i])) + ".npy", label[positiveList[i]][0]])

    for i in range(trainNum2, negativeList.shape[0]):
        if label[negativeList[i]][0] != 0:
            print("YYY")
        valList.append(
            [os.path.join(dir, "Array", str(negativeList[i])) + ".npy", label[negativeList[i]][0]])

    train = pd.DataFrame(data=trainList)
    train.to_csv(os.path.join(dir, 'train.csv'), encoding='gbk', index=False, header=False)

    test = pd.DataFrame(data=valList)
    test.to_csv(os.path.join(dir, 'val.csv'), encoding='gbk', index=False, header=False)


def generateSampleImage(NEIGHBOUR, data1, index1, index2, name):
    dir = os.path.join(OUTPUT_DIR, str(2000), str(NEIGHBOUR))
    newArray = np.zeros((data1.shape[0], data1.shape[1]), dtype="uint8")
    newArray = newArray.reshape((data1.shape[0] * data1.shape[1]))
    for i in range(len(index1)):
        newArray[index2[index1[i]]] = 255
    newArray = newArray.reshape((data1.shape[0], data1.shape[1]))
    for i in range(newArray.shape[0]):
        for j in range(newArray.shape[1]):
            if newArray[i][j] == 255:
                newArray[i - 2:i, j - 2:j] = 255
    img = Image.fromarray(newArray)
    img.save(os.path.join(dir, name + '.png'))


if __name__ == '__main__':
    dataWuhanRaw = loadGridData(DATA_DIR + "Wuhan.tif")
    dataLandYear2 = np.array(Image.open(ROOT_DIR + "2010.png"))
    dataLandYear2 = np.array(dataLandYear2 / 125, dtype=np.uint8)

    dataLandYear1 = np.array(Image.open(ROOT_DIR + "2000.png"))
    dataLandYear1 = np.array(dataLandYear1 / 125, dtype=np.uint8)

    NEIGHBOUR = 9
    RADIUS = int((NEIGHBOUR - 1) / 2)
    nonUrbanIndex_5, noWaterArray, waterArray, nonUrbanArray_5, _ = extractCityFeature(dataWuhanRaw, dataLandYear1,
                                                                                       RADIUS)
    landChangeArrays = extractChangeArea(dataLandYear2, dataLandYear1, nonUrbanArray_5)
    randomListPositive_5, randomListNegative_5 = systematic_sample_prevalences_2(landChangeArrays,
                                                                                 nonUrbanIndex_5,
                                                                                 nonUrbanIndex_5.shape[0] * 0.05,
                                                                                 0.5)

    randomListPositive_5 = np.unique(randomListPositive_5)
    randomListNegative_5 = np.unique(randomListNegative_5)
    np.random.shuffle(randomListPositive_5)
    np.random.shuffle(randomListNegative_5)

    randomListPositive5_2 = nonUrbanIndex_5[randomListPositive_5]
    randomListNegative5_2 = nonUrbanIndex_5[randomListNegative_5]
    saveCSV_trainAndval_prevalences(landChangeArrays.ravel().reshape((-1, 1))[nonUrbanIndex_5], str(NEIGHBOUR),
                                    randomListPositive_5, randomListNegative_5)
    print(randomListPositive_5.shape, randomListNegative_5.shape, (np.unique(randomListPositive_5)).shape,
          (np.unique(randomListNegative_5)).shape)

