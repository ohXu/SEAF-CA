from CNN.LoadData import *
from sklearn import preprocessing

NEIGHBOUR = 9
RADIUS = int((NEIGHBOUR - 1) / 2)
CHANNEL = 10 + 2 + 2
ROOT_DIR = "./DATA/Wuhan_F/LU/"
DATA_DIR = "./DATA/"
OLD_YEAR = str(2010)
NEW_YEAR = str(2020)
OUTPUT_DIR = os.path.join("./DATA/CA_DATA/", OLD_YEAR)


def extractNeighbourFeature(array, landData, indexArray, neighbour, space, channel, label):
    ans = []
    id = 0
    dir = os.path.join(OUTPUT_DIR, str(NEIGHBOUR))
    if not os.path.exists(dir):
        os.makedirs(dir)
        os.makedirs(os.path.join(dir, "Array"))

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if indexArray[i][j] != 0:
                if id % 1000 == 0:
                    print(id)
                data = np.zeros((neighbour, neighbour, channel + 1), dtype=np.float64)
                data[:, :, :-1] = 1 - array[i - space: i + space + 1, j - space: j + space + 1, :]
                data[:, :, -1] = landData[i - space: i + space + 1, j - space: j + space + 1] / 2

                np.save(os.path.join(dir, "Array", str(id)), data)
                ans.append([os.path.join(dir, "Array", str(id)) + ".npy", label[id][0]])
                id = id + 1

    test = pd.DataFrame(data=ans)
    test.to_csv(os.path.join(dir, 'test.csv'), encoding='gbk', index=False, header=False)


def saveCSV_trainAndval_prevalences(label, positiveList, negativeList):
    trainNum1 = int(positiveList.shape[0] * 0.90)
    trainNum2 = int(negativeList.shape[0] * 0.90)
    dir = os.path.join(OUTPUT_DIR, str(NEIGHBOUR))
    trainList = []
    valList = []
    for i in range(trainNum1):
        trainList.append(
            [os.path.join(dir, "Array", str(positiveList[i])) + ".npy", label[positiveList[i]][0]])

    for i in range(trainNum2):
        trainList.append(
            [os.path.join(dir, "Array", str(negativeList[i])) + ".npy", label[negativeList[i]][0]])

    for i in range(trainNum1, positiveList.shape[0]):
        valList.append(
            [os.path.join(dir, "Array", str(positiveList[i])) + ".npy", label[positiveList[i]][0]])

    for i in range(trainNum2, negativeList.shape[0]):
        valList.append(
            [os.path.join(dir, "Array", str(negativeList[i])) + ".npy", label[negativeList[i]][0]])

    train = pd.DataFrame(data=trainList)
    train.to_csv(os.path.join(dir, 'train.csv'), encoding='gbk', index=False, header=False)

    test = pd.DataFrame(data=valList)
    test.to_csv(os.path.join(dir, 'val.csv'), encoding='gbk', index=False, header=False)


def generateSampleImage(data1, index1, index2):
    dir = os.path.join(OUTPUT_DIR, str(NEIGHBOUR))
    newArray = np.zeros((data1.shape[0], data1.shape[1]), dtype="uint8")
    newArray = newArray.reshape((data1.shape[0] * data1.shape[1]))
    for i in range(len(index1)):
        newArray[index2[index1[i]]] = 255
    newArray = newArray.reshape((data1.shape[0], data1.shape[1]))
    img = Image.fromarray(newArray)
    img.save(os.path.join(dir, 'positive_distrobution.png'))


if __name__ == '__main__':
    dataWuhanRaw = loadGridData(DATA_DIR + "Wuhan.tif")
    dataLandYear1 = np.array(Image.open(ROOT_DIR + OLD_YEAR + ".png"))
    dataLandYear2 = np.array(Image.open(ROOT_DIR + NEW_YEAR + ".png"))
    dataLandYear1 = np.array(dataLandYear1 / 125, dtype=np.uint8)
    dataLandYear2 = np.array(dataLandYear2 / 125, dtype=np.uint8)

    proximityFactors = loadProximityDataEachYear(2020, DATA_DIR)
    natureFactors = loadNatureData(DATA_DIR)
    positionFactors = generatePositionData(dataWuhanRaw.shape[0], dataWuhanRaw.shape[1])
    allFactors = np.concatenate((proximityFactors, natureFactors), axis=-1)
    allFactors = np.concatenate((allFactors, positionFactors), axis=-1)
    allFactors2D = allFactors.reshape((allFactors.shape[0] * allFactors.shape[1], -1))

    minMaxScaler = preprocessing.MinMaxScaler()
    allFactors2D = minMaxScaler.fit_transform(allFactors2D)
    allFactors = allFactors2D.reshape((allFactors.shape[0], allFactors.shape[1], -1))

    nonUrbanIndex, noWaterArray, waterArray, nonUrbanArray, _ = extractCityFeature(dataWuhanRaw, dataLandYear1, RADIUS)
    landChange = extractChangeArea(dataLandYear2, dataLandYear1, nonUrbanArray)
    landChange = landChange.ravel().reshape((-1, 1))[nonUrbanIndex]
    extractNeighbourFeature(allFactors, dataLandYear1, nonUrbanArray, NEIGHBOUR, RADIUS, CHANNEL, landChange)
