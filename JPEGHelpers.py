import numpy as np
import math as math
import Constants as const
import RunlengthEncode


def arithmeticMean(arr, x, y, z, offsetX, offsetY):
    mean = 0
    for i in range(x, x + offsetX):
        for j in range(y, y + offsetY):
            mean += arr[j][i][z]
    return mean / (offsetX * offsetY)


def ChannelDCT(channelArray):
    height = channelArray.shape[0]
    width = channelArray.shape[1]

    n1 = math.sqrt(1.0 / const.BlockSize)
    n2 = math.sqrt(2.0 / const.BlockSize)

    result = np.zeros((height, width))
    print(result.shape)
    # Call the DCT Function FOR EACH block
    for y_block in range(0, height, const.BlockSize):
        for x_block in range(0, width, const.BlockSize):
            BlockDCT(channelArray, y_block, x_block, result, n1, n2)
    return result


def BlockDCT(channelArray, y_block, x_block, result, n1, n2):
    for y in range(y_block, y_block + const.BlockSize):
        Cy = n2
        if y == y_block:
            Cy = n1
        for x in range(x_block, x_block + const.BlockSize):
            Cx = n2
            if x == x_block:
                Cx = n1
            sum = 0
            for m in range(y_block, y_block + const.BlockSize):
                for n in range(x_block, x_block + const.BlockSize):
                    sum += channelArray[m, n] * \
                           math.cos(((2.0 * (m - y_block) + 1.0) / (2.0 * const.BlockSize)) * (y - y_block) * math.pi) * \
                           math.cos(((2.0 * (n - x_block) + 1.0) / (2.0 * const.BlockSize)) * (x - x_block) * math.pi)
            result[y, x] = sum * Cy * Cx
    return result


def ChannelDifferentialEncoding(channelArray):
    height = channelArray.shape[0]
    width = channelArray.shape[1]

    result = np.copy(channelArray)
    for y_block in range(0, height, const.BlockSize):
        difference = 0
        for x_block in range(0, width, const.BlockSize):
            new_difference = result[y_block][x_block]
            result[y_block][x_block] -= difference
            difference = new_difference
    return result


def BlockZickZack(channelArray, y_block, x_block, offsetForBlock, zickZackIndexArray, result):
    for x in range(0, const.BlockSize):
        for y in range(0, const.BlockSize):
            indexInBlock = zickZackIndexArray[x][y] - 1
            result[offsetForBlock + indexInBlock] = channelArray[x_block + x][y_block + y]


def ChannelZickZack(channelArray):
    height = channelArray.shape[0]
    width = channelArray.shape[1]

    # print(channelArray.shape)

    zickZackIndexArray = const.ZickZackMappingForBlocksize(const.BlockSize)
    result = np.zeros([width * height])

    offsetForBlock = 0
    # for each block of image
    for y_block in range(0, height, const.BlockSize):
        for x_block in range(0, width, const.BlockSize):
            BlockZickZack(channelArray, y_block, x_block, offsetForBlock, zickZackIndexArray, result)
            offsetForBlock += const.BlockSize * const.BlockSize
    return result


def ChannelInverseZickZack(channelArray, width, height):
    # zickZackIndexArray = const.ReverseZickZackMappingForBlocksize()
    result = np.zeros([width, height])
    print(len(channelArray))
    offset = 0
    for y_block in range(0, height, const.BlockSize):
        for x_block in range(0, width, const.BlockSize):
            BlockInverseZickZack(channelArray, offset, y_block, x_block, result)
            offset += const.BlockSize * const.BlockSize
    return result


def BlockInverseZickZack(channelArray, offsetInArr, y_block, x_block, result):
    ReverseZickZackMapping = const.ReverseZickZackMappingForBlocksize()
    for i in range(0, const.BlockSize * const.BlockSize):
        x_offset = ReverseZickZackMapping[i][0]
        y_offset = ReverseZickZackMapping[i][1]
        result[x_block + x_offset, y_block + y_offset] = channelArray[offsetInArr + i]


def ChannelLengthEncode(channelArray):
    blockLength = const.BlockSize * const.BlockSize
    currentResultIndex = 0
    result = np.zeros(len(channelArray) * 2)
    for currentBlockOffset in range(0, len(channelArray), blockLength):
        currentResultIndex = BlockLengthEncode(channelArray, currentBlockOffset, currentResultIndex, result)
    return result[0:currentResultIndex]


def BlockLengthEncode(channelArray, currentBlockOffset, currentResultIndex, result):
    # Just copy the dc component to the result array, will be compressed in the huffman stage
    result[currentResultIndex] = channelArray[currentBlockOffset]
    currentResultIndex += 1
    indexInCurrentBlock = 0
    while indexInCurrentBlock < (const.BlockSize - 1):
        runLength = 0
        while ++indexInCurrentBlock <= (const.BlockLength - 1) \
                and channelArray[currentBlockOffset + indexInCurrentBlock] == 0:
            runLength += 1
            if runLength == 16:
                runLength = 0
                result[currentResultIndex] = const.LongZeroRunMarker
                currentResultIndex += 1

        if indexInCurrentBlock >= const.BlockLength:
            while result[currentResultIndex - 1] == const.LongZeroRunMarker:
                currentResultIndex -= 1
            result[currentResultIndex] = const.EndOfBlockMarker
            currentResultIndex += 1
        else:
            result[currentResultIndex] = runLength
            currentResultIndex += 1
            result[currentResultIndex] = channelArray[currentBlockOffset + indexInCurrentBlock]
            currentResultIndex += 1

    if result[currentResultIndex - 1] != const.EndOfBlockMarker:
        result[currentResultIndex] = const.EndOfBlockMarker
        currentResultIndex += 1
    return currentResultIndex


def ChannelRunlengthDecode(channelArray, blockCount):
    # blockLength is the length of an 2d square with the side length of block_size
    blockLength = const.BlockSize ** 2
    # Points to the next free index where data can be written to
    currentWriteIndex = 0
    # Points to the index which will be decoded next
    currentDecodeIndex = 0

    result = np.zeros(blockLength * blockCount)

    while blockCount > 0:
        blockCount -= 1
        # Copy DC value into the output array
        result[currentWriteIndex] = channelArray[currentDecodeIndex]
        currentDecodeIndex += 1
        currentWriteIndex += 1
        while True:
            # Check if we reach an end of block
            if channelArray[currentDecodeIndex] == RunlengthEncode.endOfBlockMarker:
                currentDecodeIndex += 1
                # skip to the next block boundary
                currentWriteIndex = ((currentWriteIndex + blockLength - 1) / blockLength) * blockLength
                break
            else:
                if channelArray[currentDecodeIndex] == RunlengthEncode.longZeroRunMarker:
                    currentDecodeIndex += 1
                    # skip over the zeros
                    currentWriteIndex += 16
                else:
                    # skip over the zeros...
                    currentWriteIndex += channelArray[currentDecodeIndex]
                    currentDecodeIndex += 1
                    # ...and add the value behind that
                    result[currentWriteIndex] = channelArray[currentDecodeIndex]
                    currentDecodeIndex += 1
                    currentWriteIndex += 1
    return result


def ChannelInverseDifferentialEncoding(channelArray):
    width = len(channelArray[0])
    height = len(channelArray)
    result = channelArray[:]
    PrevioudD = 0
    for y_block in range(0, height, const.BlockSize):
        PrevioudD = 0
        for x_block in range(0, width, const.BlockSize):
            result[y_block][x_block] += PrevioudD
            PrevioudD = result[y_block][x_block]
    return result


def ChannelQuantization(channelArray, quantMatrix, fileName):
    n = len(quantMatrix)
    width = len(channelArray[0])
    height = len(channelArray)
    result = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            result[y, x] = int(round(channelArray[y, x] / float(quantMatrix[int(y % n)][int(x % n)])))
    np.savetxt(fileName + "_quantization.txt", result, delimiter=" ", fmt="%s")
    return result


def ChannelDequantization(channelArray, quantMatrix, fileName):
    n = len(quantMatrix)
    width = len(channelArray[0])
    height = len(channelArray)
    result = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            result[y, x] = channelArray[y][x] * quantMatrix[int(y % n)][int(x % n)]
    np.savetxt(fileName + "_dequantization.txt", result, delimiter=" ", fmt="%s")
    return result


def ChannelIdct(channelArray, fileName):
    width = len(channelArray[0])
    height = len(channelArray)
    n1 = math.sqrt(1.0 / const.BlockSize)
    n2 = math.sqrt(2.0 / const.BlockSize)
    result = np.zeros((height, width))
    # for each block of image
    for y_block in range(0, height, const.BlockSize):
        for x_block in range(0, width, const.BlockSize):
            BlockIdct(channelArray, y_block, x_block, result, n1, n2)
    np.savetxt(fileName + "_idct.txt", result, delimiter=" ", fmt="%s")
    return result


def BlockIdct(channelArray, y_block, x_block, result, n1, n2):
    # for each element in block
    for x in range(y_block, y_block + const.BlockSize):
        for y in range(x_block, x_block + const.BlockSize):
            # calculate idct (over block)
            sum = 0
            for m in range(y_block, y_block + const.BlockSize):
                Cy = n1 if (m == y_block) else n2
                for n in range(x_block, x_block + const.BlockSize):
                    Cx = n1 if (n == x_block) else n2
                    sum += (Cx * Cy) * channelArray[m, n] * \
                           math.cos((2.0 * (x - y_block) + 1.0) * (m - y_block) * math.pi / (2.0 * const.BlockSize)) * \
                           math.cos((2.0 * (y - x_block) + 1.0) * (n - x_block) * math.pi / (2.0 * const.BlockSize))
            result[x, y] = sum


def ReverseSubsampling_TYPE_4_1_1(yCbCrChannels, result):
    for x in range(len(yCbCrChannels[1])):
        for y in range(len(yCbCrChannels[1][0])):
            for i in range(4):
                result[y][4 * x + i][1] = yCbCrChannels[1][y][x]
                result[y][4 * x + i][2] = yCbCrChannels[2][y][x]


def ReverseSubsampling_TYPE_4_2_0(yCbCrChannels, result):
    for x in range(len(yCbCrChannels[1])):
        for y in range(len(yCbCrChannels[1][0])):
            for i in range(2):
                for j in range(2):
                    result[2 * y + i][2 * x + j][1] = yCbCrChannels[1][y][x]
                    result[2 * y + i][2 * x + j][2] = yCbCrChannels[2][y][x]


def ReverseSubsampling_TYPE_4_2_2(yCbCrChannels, result):
    for x in range(len(yCbCrChannels[1])):
        for y in range(len(yCbCrChannels[1][0])):
            for i in range(2):
                result[y][2 * x + i][1] = yCbCrChannels[1][y][x]
                result[y][2 * x + i][2] = yCbCrChannels[2][y][x]


def ReverseSubsampling_TYPE_4_4_4(yCbCrChannels, result):
    for x in range(len(yCbCrChannels[1])):
        for y in range(len(yCbCrChannels[1][0])):
            result[y][x][1] = yCbCrChannels[1][y][x]
            result[y][x][2] = yCbCrChannels[2][y][x]


def mapit(val):
    if val > 255:
        return 255
    if val < 0:
        return 0
    return int(val)


def minus(a, b):
    result = np.zeros((len(a), len(b[0])))
    for i in range(len(a)):
        for j in range(len(b[0])):
            result[i][j] = a[i][j] - b[i][j]
    return result


def mult(a, b):
    result = np.zeros((len(a), len(b[0])))
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                result[i][j] = result[i][j] + a[i][k] * b[k][j]
    return result
