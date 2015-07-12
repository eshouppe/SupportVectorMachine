import random
import numpy as np


def loadDataSet(filename):
    # Opens the file and parses each line into class labels and data matrix
    dataMat, labelMat = [], []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat, labelMat


def selectRandomJ(k, m):
    # Variable i is the index of the first alpha.  Variable m is the total number of alphas.
    # Randomly choose a an alpha value by its index and return it if it != i.
    j = k
    while j==k:
        j = int(random.uniform(0,m))
    return j


def clipAlpha(alphaJ, high, low):
    # Clip alpha value so it is within [low, high] range.
    if alphaJ > high:
        alphaJ = high

    if alphaJ < low:
        alphaJ = low

    return alphaJ


def simpleSMO(datMatIn, classLabel, c, toler, maxIter):
    # SMO (Sequential Minimal Optimization) algorithm finds a set of alphas and the constant b
    dataMatrix = np.mat(datMatIn)
    labelMatrix = np.mat(classLabel).transpose()
    counter, b = 0, 0
    rowCnt = np.shape(dataMatrix)[0]
    alphas = np.mat(np.zeros((rowCnt,1)))

    while counter < maxIter:
        alphaPairsChanged = 0
        for i in range(rowCnt):
            # Prediction of the class. f(x) = w^T * x + b, which is also f(x) = alphas * labels * <xi, x> + b
            fOfXi = float(np.multiply(alphas,labelMatrix).T * (dataMatrix*dataMatrix[i,:].T)) + b
            # Error based on prediction and real instance of class
            errorI = fOfXi - float(labelMatrix[i])

            # If the errorI is large, the alphas corresponding to the real instance are optimized
            # Check to see is alpha equals C or 0, bc ones that are cannot be optimized
            if ((labelMatrix[i]*errorI<-toler)and(alphas[i]<c))or((labelMatrix[i]*errorI>toler)and(alphas[i]>0)):
                # Randomly select a second alpha.  Maximize alpha[i] and alpha[j] to maximize the objective function
                j = selectRandomJ(i, rowCnt)
                fOfXj = float(np.multiply(alphas,labelMatrix).T * (dataMatrix*dataMatrix[j,:].T)) + b
                # Calculate the error of the second alpha
                errorJ = fOfXj - float(labelMatrix[j])
                # old alpha [i] and [j] prior to optimization
                oldAlphaI = alphas[i].copy()
                oldAlphaJ = alphas[j].copy()

                # alpha[i] and alpha[j] are the Lagrange multipliers to optimize.  Limits are selected such that
                # low <= alpha[j] <= high to satisfy the constraint 0 <= alpha[j] <= C.
                if labelMatrix[i] != labelMatrix[j]:
                    lowLimit = max(0, alphas[j] - alphas[i])
                    highLimit = min(c, c+alphas[j] - alphas[i])
                else:
                    lowLimit = max(0, alphas[i] + alphas[j] - c)
                    highLimit = min(c, alphas[i] + alphas[j])

                # If low and high limits equal, exit loop because alpha[j] cannot be optimized
                if lowLimit == highLimit:
                    continue

                # Eta is used to calculate the optimal amount to change alpha[j]
                eta = (2.0 * dataMatrix[i,:] * dataMatrix[j,:].T)
                eta -= ((dataMatrix[i,:] * dataMatrix[i,:].T) + (dataMatrix[j,:] * dataMatrix[j,:].T))

                # If eta equals 0, exit loop because alpha[j] cannot be optimized
                if eta >= 0:
                    continue

                alphas[j] -= labelMatrix[j] * (errorI-errorJ) / eta
                # Clip alpha[j] to ensure low <= alpha[j] <= high
                alphas[j] = clipAlpha(alphas[j], highLimit, lowLimit)

                if abs(alphas[j] - oldAlphaJ) < 0.00001:
                    continue

                # alpha[i] changed in opposite direction from alpha[j]
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (oldAlphaJ - alphas[j])

                # Set the constant term for this pair of optimized alphas such that the KKT conditions are satisfied
                b1 = b - errorI - ((alphas[i] - oldAlphaI) * labelMatrix[i] * dataMatrix[i,:] * dataMatrix[i,:].T)
                b1 -= ((alphas[j] - oldAlphaJ) * labelMatrix[j] * dataMatrix[i,:] * dataMatrix[j,:].T)

                b2 = b - errorJ - ((alphas[i] - oldAlphaI) * labelMatrix[i] * dataMatrix[i,:] * dataMatrix[j,:].T)
                b2 -= ((alphas[j] - oldAlphaJ) * labelMatrix[j] * dataMatrix[j,:] * dataMatrix[j,:].T)

                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b= b2
                else:
                    # If both optimized alphas are at 0/c, than all thresholds between b1 & b2 satisfy KKT conditions
                    b = (b1 + b2) / 2.0

                # No continue statements caused exit from loop so the pair of alphas are optimized
                alphaPairsChanged += 1

        # Will only exit while loop when entire date set has be traversed maxIter number of times without change.
        if alphaPairsChanged == 0:
            counter += 1
        else:
            counter = 0

    return b, alphas


filepath = 'SupportVectorMachine/dataSVM.txt'
dataArr,labelArr = loadDataSet(filepath)
# Regularization parameter c can be tuned for different results.
constantB, alphasOptimized = simpleSMO(dataArr, labelArr, 0.6, 0.001, 20)
# Output: alphas are the Lagrange multipliers for the solution and b is the threshold for the solution
