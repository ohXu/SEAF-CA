
import numpy as np
from CNN.LoadData import *


def getNeighbourhoodEffect(data, indexArray, radius):

    weights = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            d = math.sqrt(math.pow((i - 2), 2) + math.pow((j - 2), 2))
            weights[i, j] = 1 - d / (math.sqrt(8)) / 2
    weights[2, 2] = 1
    print(weights)

    row = data.shape[0]
    col = data.shape[1]
    neighbourhoodMap = np.zeros((row, col))
    index = np.where(data == 2)
    data1 = np.zeros((row, col))
    for i in range(index[0].shape[0]):
        data1[index[0][i], index[1][i]] = 1

    for i in range(row):
        for j in range(col):
            if indexArray[i][j] == 255:
                k = np.array(data[i - radius: i + radius + 1, j - radius: j + radius + 1])
                k1 = np.array(data1[i - radius: i + radius + 1, j - radius: j + radius + 1])
                k1 = k1 * weights
                number1 = np.where(k == 2)[0].shape[0]
                # if (number1 != np.sum(k1)):
                #     print(number1, np.sum(k1))
                neighbourhoodMap[i][j] = np.sum(k1)
    radius = radius * 2 + 1
    neighbourhoodMap = neighbourhoodMap / (radius * radius - 1)
    return neighbourhoodMap


def getNeighbourhoodEffectAndConstraint(data, indexArray, radius):
    row = data.shape[0]
    col = data.shape[1]
    neighbourhoodMap = np.zeros((row, col))
    constraintMap = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            if indexArray[i][j] == 255:
                if data[i][j] == 0:
                    constraintMap[i][j] = 1
                else:
                    print("Y1")
                    constraintMap[i][j] = 0
                k = np.array(data[i - radius: i + radius + 1, j - radius: j + radius + 1])
                number1 = np.where(k == 2)[0].shape[0]
                if k[radius, radius] == 2:
                    print("Y2")
                    number1 = number1 - 1
                neighbourhoodMap[i][j] = number1
    radius = radius * 2 + 1
    neighbourhoodMap = neighbourhoodMap / (radius * radius - 1)
    print(np.max(neighbourhoodMap), np.min(neighbourhoodMap), "m")
    data_new_img = np.array(neighbourhoodMap * 255, dtype=np.uint8)
    img = Image.fromarray(data_new_img)
    img.save(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), "nei.png"))
    return neighbourhoodMap, constraintMap


def getConstraint(data, indexArray):
    row = data.shape[0]
    col = data.shape[1]
    constraintMap = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if indexArray[i][j] == 255:
                if data[i][j] == 0:
                    constraintMap[i][j] = 1
                else:
                    constraintMap[i][j] = 0
    return constraintMap


def getPotential(indexArray, predict, dir):
    im_height = indexArray.shape[0]
    im_width = indexArray.shape[1]
    potentialMap = np.zeros((im_height, im_width))
    n_helper = 0
    for row in range(im_height):
        for col in range(im_width):
            if indexArray[row][col] == 255:
                potentialMap[row][col] = predict[n_helper]
                n_helper += 1
    print("getPotential", n_helper)
    print(np.max(potentialMap), np.min(potentialMap), "m")
    data_new_img = np.array(potentialMap * 255, dtype=np.uint8)
    img = Image.fromarray(data_new_img)
    img.save(os.path.join(dir, "tp.png"))
    return potentialMap


def getStochastic(indexArray):
    row = indexArray.shape[0]
    col = indexArray.shape[1]
    stochasticMap = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if indexArray[i][j] == 255:
                stochasticMap[i][j] = 1 + math.pow((-1 * math.log(math.e, random.random())), 0.05)
    return stochasticMap


def generateImage(data, dir):
    img = np.zeros((data.shape[0], data.shape[1], 3), dtype="uint8")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 1:
                img[i, j, :] = [0, 255, 0]
            if data[i][j] == 2:
                img[i, j, :] = [0, 0, 255]
            if data[i][j] == 3:
                img[i, j, :] = [255, 0, 0]

    img = Image.fromarray(img)
    img.save(os.path.join(dir, str(NEIGHBOUR), "simulateImage.png"))


if __name__ == '__main__':
    NEIGHBOUR = 9
    RADIUS = int((NEIGHBOUR - 1) / 2)
    ROOT_DIR = "./DATA/Wuhan_F/LU/"
    DATA_DIR = "./DATA/"
    INITIAL_YEAR = str(2000)
    OLD_YEAR = str(2000)
    NEW_YEAR = str(2010)
    OUTPUT_DIR2 = os.path.join("./DATA/CA_DATA/", INITIAL_YEAR)

    dataWuhanRaw = loadGridData(DATA_DIR + "Wuhan.tif")
    dataLandYear0 = np.array(Image.open(ROOT_DIR + INITIAL_YEAR + ".png"))
    dataLandYear0 = np.array(dataLandYear0 / 125, dtype=np.uint8)
    dataLandYear2 = np.array(Image.open(ROOT_DIR + NEW_YEAR + ".png"))
    dataLandYear2 = np.array(dataLandYear2 / 125, dtype=np.uint8)

    nonUrbanIndex0, noWaterArray0, waterArray0, nonUrbanArray0, inIndex0 = extractCityFeature(dataWuhanRaw,
                                                                                              dataLandYear0,
                                                                                              RADIUS)

    nonUrbanIndex2, noWaterArray2, waterArray2, nonUrbanArray2, inIndex2 = extractCityFeature(dataWuhanRaw,
                                                                                              dataLandYear2,
                                                                                              RADIUS)

    expandRate = getExpandRate(dataLandYear2.ravel()[nonUrbanIndex0], dataLandYear0.ravel()[nonUrbanIndex0])
    print("Markov:", expandRate)

    neighbourhood, restraint = getNeighbourhoodEffectAndConstraint(dataLandYear0, nonUrbanArray0, RADIUS)
    # neighbourhood = getNeighbourhoodEffect(dataLandYear0, nonUrbanArray0, RADIUS)
    transitionPotential = np.load(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), OLD_YEAR + "pre.npy"))
    transitionPotential = getPotential(nonUrbanArray0, transitionPotential, os.path.join(OUTPUT_DIR2, str(NEIGHBOUR)))
    stochasticPerturbation = getStochastic(nonUrbanArray0)

    totalProbability = transitionPotential
    totalProbability = np.multiply(totalProbability, restraint)
    totalProbability = np.multiply(totalProbability, stochasticPerturbation)

    totalProbability_sort0 = np.sort(totalProbability.ravel())
    threshold0 = int(nonUrbanIndex0.shape[0] * expandRate)
    print("threshold0:", threshold0)
    threshold0 = totalProbability_sort0[-threshold0]
    print("threshold0:", threshold0, totalProbability_sort0[-1])

    simulateLand1 = np.zeros((dataLandYear2.shape[0], dataLandYear2.shape[1]))
    simulateLand0 = np.zeros((dataLandYear0.shape[0], dataLandYear0.shape[1]))
    for i in range(dataLandYear0.shape[0]):
        for j in range(dataLandYear0.shape[1]):
            if noWaterArray0[i][j] == 255:
                simulateLand0[i][j] = dataLandYear0[i][j] + 1
                if totalProbability[i][j] >= threshold0 and simulateLand0[i][j] == 1:
                    simulateLand0[i][j] = 3
                    simulateLand1[i][j] = 255

            if waterArray0[i][j] == 255:
                simulateLand0[i][j] = 2

    np.save(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), "tp"), transitionPotential)
    np.save(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), "tp_c"), totalProbability)
    np.save(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), "nonUrbanIndex"), nonUrbanIndex0)
    np.save(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), "inIndex"), inIndex0)
    np.save(os.path.join(OUTPUT_DIR2, str(NEIGHBOUR), "simulateArray"), simulateLand0)
    generateImage(simulateLand0, OUTPUT_DIR2)