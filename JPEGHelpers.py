import numpy as np
import math as math

def arithmeticMean(arr, x, y, z, offsetX, offsetY):
    mean = 0
    for i in range(x, x + offsetX):
        for j in range(y, y + offsetY):
            mean += arr[j][i][z]
    return mean / (offsetX * offsetY)


def ChannelDCT(channelArray, blockSize):
    height = channelArray.shape[0]
    width = channelArray.shape[1]

    n1 = math.sqrt(1.0 / blockSize)
    n2 = math.sqrt(2.0 / blockSize)

    result = [np.zeros(height)[:], np.zeros(width)[:]]

    # Call the DCT Function FOR EACH block
    for y_block in range(0, height, blockSize):
        for x_block in range(0, width, blockSize):
            BlockDCT(channelArray, y_block, x_block, result, blockSize, n1, n2)
    return result


def BlockDCT(channelArray, y_block, x_block, result, blockSize, n1, n2):
    for y in range(y_block, y_block + blockSize):
        Cy = n2
        if y == y_block:
            Cy = n1
        for x in range(x_block, x_block + blockSize):
            Cx = n2
            if x == x_block:
                Cx = n1
            sum = 0
            for m in range(y_block, y_block + blockSize):
                for n in range(x_block, x_block + blockSize):
                    sum += channelArray[m][n] * \
                        math.cos(((2.0 * (m - y_block) + 1.0) / (2.0 * blockSize)) * (y - y_block) * math.pi) * \
                        math.cos(((2.0 * (n - x_block) + 1.0) / (2.0 * blockSize)) * (x - x_block) * math.pi)
            result[y][x] = sum * Cy * Cx
    return result

