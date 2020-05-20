import numpy as np
import math
import random


############## Optimal Codebook Generation ####################

def makeBook(base, m, n):
    book = makeSquareBook(base, math.ceil(math.log(max(m, n), base)))[-m:, -n:]
    if base == 2:
        book[0, 0] = 0 if (book[0, 0] == 1) else 1
    return book


def makeSquareBook(base, expansions):
    startBook = getStartBook(base)
    fullBook = np.copy(startBook)
    for i in range(expansions):
        fullBook = expand(startBook, fullBook)
    return fullBook


def expand(baseBook, bookToExpand):
    rotations = [(bookToExpand+i) % len(baseBook)
                 for i in range(len(baseBook))]
    # very hard to read, but would require lots of extra effort to improve readability
    # This line simply replaces each element of baseBook with rotations[element]
    return np.concatenate(list(np.concatenate(list(
        rotations[rotationAmt] for rotationAmt in baseBook[row]), axis=1) for row in range(len(baseBook))), axis=0)


def getStartBook(base):
    if base % 2 is 1:
        topRow = []
        element = 0
        for i in reversed(range(1, base+1)):
            topRow.append(element)
            element = (element+i) % base
        topRow = np.array(topRow)
        book = np.array([topRow])
        for i in range(1, base):
            book = np.append(
                book, [(np.append(topRow[-i:], topRow[:-i])+i) % base], axis=0)
        return book
    elif base > 1:
        book = getStartBook(base-1)
        book = np.append([[base-1]]*(base-1), book, axis=1)
        return np.append([[base-1]*(base)], book, axis=0)
    else:
        return 0


############## Special Matrix Generation ##############


def hadamardWithNumTernaryPerLine(m, n, numTernary):
    book = makeBook(2, m, n)
    random.seed(0)
    for row in book:
        for index in random.sample(range(m), numTernary):
            row[index] = 2
    return book


def makeRandomBook(base, m, n):
    return np.random.randint(0, base, (m, n))


STATUS_OPTIMAL = 0
STATUS_RANDOM = 1
STATUS_BEST_HAMMING_OF_1000 = 2
STATUS_BEST_ABSOLUTE_OF_1000 = 3


def makeBookWithInfo(base, m, n, status):
    info = {}
    if status is STATUS_OPTIMAL:
        info["book"] = makeBook(base, m, n)
    elif status is STATUS_RANDOM:
        info["book"] = makeRandomBook(base, m, n)
    elif status is STATUS_BEST_HAMMING_OF_1000:
        info["book"] = bestMinHammingDistOfRandoms(base, m, n, 1000)
    elif status is STATUS_BEST_ABSOLUTE_OF_1000:
        info["book"] = bestMinAbsoluteDistOfRandoms(base, m, n, 1000)
    else:
        info["book"] = None
        return info
    info["width"] = n
    info["height"] = m
    info["base"] = base

    minRowDist = n
    maxRowDist = 0
    totalRowDist = 0
    totalRowSpace = 0
    numDistances = 0
    for i, row1 in enumerate(info["book"][:-1]):
        for row2 in info["book"][i+1:]:
            dist = getHammingDist(row1, row2)
            if dist < minRowDist:
                minRowDist = dist
            if dist > maxRowDist:
                maxRowDist = dist
            totalRowDist += dist
            totalRowSpace += abs(dist - n + n/base)
            numDistances += 1
    info["Minimum Row Hamming Distance"] = minRowDist
    info["Maximum Row Hamming Distance"] = maxRowDist
    info["Average Row Hamming Distance"] = totalRowDist / numDistances
    info["Average Row Hamming Space"] = totalRowSpace / numDistances

    transpose = np.transpose(info["book"])
    minColDist = m
    maxColDist = 0
    totalColDist = 0
    totalColSpace = 0
    numDistances = 0
    for i, col1 in enumerate(transpose[:-1]):
        for col2 in transpose[i+1:]:
            dist = getHammingDist(col1, col2)
            if dist < minColDist:
                minColDist = dist
            if dist > maxColDist:
                maxColDist = dist
            totalColDist += dist
            totalColSpace += abs(dist - m + m/base)
            numDistances += 1
    info["Minimum Column Hamming Distance"] = minColDist
    info["Maximum Column Hamming Distance"] = maxColDist
    info["Average Column Hamming Distance"] = totalColDist / numDistances
    info["Average Column Hamming Space"] = totalColSpace / numDistances

    minRowDist = n
    maxRowDist = 0
    totalRowDist = 0
    totalRowSpace = 0
    numDistances = 0
    for i, row1 in enumerate(info["book"][:-1]):
        for row2 in info["book"][i+1:]:
            dist = getAbsoluteDist(row1, row2)
            if dist < minRowDist:
                minRowDist = dist
            if dist > maxRowDist:
                maxRowDist = dist
            totalRowDist += dist
            totalRowSpace += abs(dist - n + n/base)
            numDistances += 1
    info["Minimum Row Absolute Distance"] = minRowDist
    info["Maximum Row Absolute Distance"] = maxRowDist
    info["Average Row Absolute Distance"] = totalRowDist / numDistances
    info["Average Row Absolute Space"] = totalRowSpace / numDistances

    transpose = np.transpose(info["book"])
    minColDist = m
    maxColDist = 0
    totalColDist = 0
    totalColSpace = 0
    numDistances = 0
    for i, col1 in enumerate(transpose[:-1]):
        for col2 in transpose[i+1:]:
            dist = getAbsoluteDist(col1, col2)
            if dist < minColDist:
                minColDist = dist
            if dist > maxColDist:
                maxColDist = dist
            totalColDist += dist
            totalColSpace += abs(dist - m + m/base)
            numDistances += 1
    info["Minimum Column Absolute Distance"] = minColDist
    info["Maximum Column Absolute Distance"] = maxColDist
    info["Average Column Absolute Distance"] = totalColDist / numDistances
    info["Average Column Absolute Space"] = totalColSpace / numDistances

    return info


############## Additional Functions #################


def getAbsoluteDist(row1, row2):
    dist = 0
    for i in range(row1.size):
        dist += abs(row1[i] - row2[i])
    return dist


def getMinAbsoluteDist(book):
    mindist = book.size
    total = 0
    for i, row1 in enumerate(book[:-1]):
        for row2 in book[i+1:]:
            dist = getAbsoluteDist(row1, row2)
            total += dist
            mindist = min(dist, mindist)
    return mindist


def getHammingDist(row1, row2):
    dist = 0
    for i in range(row1.size):
        if row1[i] != row2[i]:
            dist = dist + 1
    return dist


def getMinHammingDist(book):
    mindist = book.size
    total = 0
    for i, row1 in enumerate(book[:-1]):
        for row2 in book[i+1:]:
            dist = getHammingDist(row1, row2)
            total += dist
            mindist = min(dist, mindist)
    return mindist


def bestMinHammingDistOfRandoms(base, m, n, numCodebooks):
    bestCodebook = makeRandomBook(base, m, n)
    bestMinDist = getMinHammingDist(
        bestCodebook) + getMinHammingDist(np.transpose(bestCodebook))
    for i in range(numCodebooks - 1):
        codebook = makeRandomBook(base, m, n)
        minDist = getMinHammingDist(
            codebook) + getMinHammingDist(np.transpose(codebook))
        if (minDist > bestMinDist):
            bestCodebook = codebook
            bestMinDist = minDist
    return bestCodebook


def bestMinAbsoluteDistOfRandoms(base, m, n, numCodebooks):
    bestCodebook = makeRandomBook(base, m, n)
    bestMinDist = getMinAbsoluteDist(
        bestCodebook) + getMinAbsoluteDist(np.transpose(bestCodebook))
    for i in range(numCodebooks - 1):
        codebook = makeRandomBook(base, m, n)
        minDist = getMinAbsoluteDist(
            codebook) + getMinAbsoluteDist(np.transpose(codebook))
        if (minDist > bestMinDist):
            bestCodebook = codebook
            bestMinDist = minDist
    return bestCodebook


def printBookInfoForSpreadsheet(info):
    print(info["book"].tolist())
    for element in list(info)[1:]:
        print(info[element])


printBookInfoForSpreadsheet(makeBookWithInfo(
    23, 27, 27, STATUS_BEST_HAMMING_OF_1000))
