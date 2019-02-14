import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Constants import *
from HuffmanDecoding import HuffmanDecoding
from HuffmanEncoding import HuffmanEncoding
from JPEGHelpers import *
import HuffmanTree


def step1_LoadImage(debugFlag, imagePath):
    print("Step 1 Loading Image", imagePath)
    img = Image.open(imagePath)
    imageArray = np.array(img)  # 640x480x4 array
    if debugFlag:
        print(imageArray.shape)
        plt.imshow(imageArray)
        plt.show()
    return imageArray


def step2_ConvertRGBToYCbCr(debugFlag, rgbChannels):
    print("Step 2 Converting to YCbCr")
    height = rgbChannels.shape[0]
    width = rgbChannels.shape[1]
    yCbCrPicture = np.zeros((height, width, 3))

    # FOR EACH RGB-PIXEL
    for x in range(width):
        for y in range(height):
            # FLIP: R G B TO R
            #                G
            #                B
            # This is surely not really needed^^
            rgbVector = [
                [rgbChannels[x, y, 0]],  # RED-Value of the current Pixel
                [rgbChannels[x, y, 1]],  # BLUE-Value of the current Pixel
                [rgbChannels[x, y, 2]],  # GREEN-Value of the current Pixel
            ]
            ypbpr = np.matrix(MultiplicationMatrixYCbCr) * rgbVector
            ycbcr = np.matrix(ypbpr) + AdditionMatrixYCbCr
            yCbCrPicture[x, y, 0] = ycbcr[0, 0]  # Y
            yCbCrPicture[x, y, 1] = ycbcr[1, 0]  # Cb
            yCbCrPicture[x, y, 2] = ycbcr[2, 0]  # Cr
    if debugFlag:
        plt.imshow(yCbCrPicture[:, :, 0])
        plt.xlabel('For Y')
        plt.set_cmap('gray')
        plt.show()

        plt.imshow(yCbCrPicture[:, :, 1])
        plt.xlabel('For Cb')
        plt.set_cmap('gray')
        plt.show()

        plt.imshow(yCbCrPicture[:, :, 2])
        plt.xlabel('For Cr')
        plt.set_cmap('gray')
        plt.show()
    return yCbCrPicture


def step3_SubSample(debugFlag, yCbCrChannels, sampleOverX: int, sampleOverY: int):
    print("Step 3 Subsample")
    height = yCbCrChannels.shape[0]
    width = yCbCrChannels.shape[1]
    result = None
    found = False
    # 4 Different Modes for Sub-Sample
    if sampleOverX == 1 and sampleOverY == 1:
        result = Subsampling_TYPE_4_1_1(yCbCrChannels, height, width)
        found = True
    if sampleOverX == 2 and sampleOverY == 0:
        result = Subsampling_TYPE_4_2_0(yCbCrChannels, height, width)
        found = True
    if sampleOverX == 2 and sampleOverY == 2:
        result = Subsampling_TYPE_4_2_2(yCbCrChannels, height, width)
        found = True
    if sampleOverX == 4 and sampleOverY == 4:
        result = Subsampling_TYPE_4_4_4(yCbCrChannels, height, width)
        found = True
    if found:
        result = [yCbCrChannels[:, :, 0], result[:, :, 1], result[:, :, 2]]
        if debugFlag:
            plt.imshow(result[0])
            plt.xlabel('ReverseSubsampling For Y')
            plt.set_cmap('gray')
            plt.show()

            plt.imshow(result[1])
            plt.xlabel('ReverseSubsampling For Cb')
            plt.set_cmap('gray')
            plt.show()

            plt.imshow(result[2])
            plt.xlabel('ReverseSubsampling For Cr')
            plt.set_cmap('gray')
            plt.show()
        return result
    else:
        raise ValueError("Wrong sample type")
    # channelCb = np.zeros([int(height / sampleOverY), int(width / sampleOverX)])
    # channelCr = np.zeros([int(height / sampleOverY), int(width / sampleOverX)])
    # for x in range(0, width, sampleOverX):
    #     for y in range(0, height, sampleOverY):
    #         channelCb[int(y / sampleOverY), int(x / sampleOverX)] = arithmeticMean(yCbCrChannels, x, y, 1, sampleOverX,
    #                                                                                sampleOverY)
    #         channelCr[int(y / sampleOverY), int(x / sampleOverX)] = arithmeticMean(yCbCrChannels, x, y, 2, sampleOverX,
    #                                                                                sampleOverY)
    # if debugFlag:
    #     plt.imshow(channelY)
    #     plt.xlabel('For Y AFTER')
    #     plt.set_cmap('gray')
    #     plt.show()
    #
    #     plt.imshow(channelCb)
    #     plt.xlabel('For Cb')
    #     plt.set_cmap('gray')
    #     plt.show()
    #
    #     plt.imshow(channelCr)
    #     plt.xlabel('For Cr')
    #     plt.set_cmap('gray')
    #     plt.show()
    return [channelY, channelCb, channelCr]


def step4_DCTAllChannels(debugFlag, yCbCrChannels):
    print("Step 4 DCT Channels")
    dctY = ChannelDCT(yCbCrChannels[0])
    print("  DCT Channel Y")
    dctCb = ChannelDCT(yCbCrChannels[1])
    print("  DCT Channel Cb")
    dctCr = ChannelDCT(yCbCrChannels[2])
    print("  DCT Channel Cr")
    if debugFlag:
        plt.imshow(dctY)
        plt.xlabel('For Y AFTER')
        plt.set_cmap('gray')
        plt.show()

        plt.imshow(dctCb)
        plt.xlabel('For Cb')
        plt.set_cmap('gray')
        plt.show()

        plt.imshow(dctCr)
        plt.xlabel('For Cr')
        plt.set_cmap('gray')
        plt.show()
    return [dctY, dctCb, dctCr]


def step5_Quantization(debugFlag, yCbCrChannels):
    print("Step 5a Quantization")
    quantY = ChannelQuantization(yCbCrChannels[0], const.Q50, "layerY")
    quantCb = ChannelQuantization(yCbCrChannels[1], const.Q50, "layerCb")
    quantCr = ChannelQuantization(yCbCrChannels[2], const.Q50, "layerCr")
    return [quantY, quantCb, quantCr]


def step5_DifferentalEncoding(debugFlag, yCbCrChannels):
    print("Step 5b DifferentalEncoding")
    dfY = ChannelDifferentialEncoding(yCbCrChannels[0])
    dfCb = ChannelDifferentialEncoding(yCbCrChannels[1])
    dfCr = ChannelDifferentialEncoding(yCbCrChannels[2])
    return [dfY, dfCb, dfCr]


def step6_ZickZack(debugFlag, yCbCrChannels):
    print("Step 6 ZickZack")
    zickZackY = ChannelZickZack(yCbCrChannels[0])
    zickZackCb = ChannelZickZack(yCbCrChannels[1])
    zickZackCr = ChannelZickZack(yCbCrChannels[2])
    # if debugFlag:
    #     plt.imshow(zickZackY)
    #     plt.xlabel('For Y AFTER')
    #     plt.set_cmap('gray')
    #     plt.show()
    #
    #     plt.imshow(zickZackCb)
    #     plt.xlabel('For Cb')
    #     plt.set_cmap('gray')
    #     plt.show()
    #
    #     plt.imshow(zickZackCr)
    #     plt.xlabel('For Cr')
    #     plt.set_cmap('gray')
    #     plt.show()
    return [zickZackY, zickZackCb, zickZackCr]


def step7_LengthEncode(debugFlag, yCbCrChannels):
    print("Step 7 LenghtEncode")
    lengthEncodeY = ChannelLengthEncode(yCbCrChannels[0])
    lengthEncodeCb = ChannelLengthEncode(yCbCrChannels[1])
    lengthEncodeCr = ChannelLengthEncode(yCbCrChannels[2])
    return [lengthEncodeY, lengthEncodeCb, lengthEncodeCr]


def step8_HuffmanEncode(debugFlag, yCbCrChannels, zicks):
    print("Step 8 HuffmanEncode")
    huffmanTreeAcCx = HuffmanTree(const.StdACChrominanceLengths, const.StdACChrominanceValues)
    huffmanTreeDcCx = HuffmanTree(const.StdDCChrominanceLengths, const.StdDCChrominanceValues)
    huffmanTreeAcY = HuffmanTree(const.StdACLuminanceLengths, const.StdACLuminanceValues)
    huffmanTreeDcY = HuffmanTree(const.StdDCLuminanceLengths, const.StdDCLuminanceValues)

    huffmanEncodeY = HuffmanEncoding(zicks[0], huffmanTreeAcY, huffmanTreeDcY)
    huffmanEncodeCb = HuffmanEncoding(zicks[1], huffmanTreeAcCx, huffmanTreeDcCx)
    huffmanEncodeCr = HuffmanEncoding(zicks[2], huffmanTreeAcCx, huffmanTreeDcCx)
    return [huffmanEncodeY.result, huffmanEncodeCb.result, huffmanEncodeCr.result]


def step9_HuffmanDecode(debugFlag, yCbCrChannels, blockCount):
    print("Step 9 HuffmanDecode")
    huffmanTreeAcCx = HuffmanTree(const.StdACChrominanceLengths, const.StdACChrominanceValues)
    huffmanTreeDcCx = HuffmanTree(const.StdDCChrominanceLengths, const.StdDCChrominanceValues)
    huffmanTreeAcY = HuffmanTree(const.StdACLuminanceLengths, const.StdACLuminanceValues)
    huffmanTreeDcY = HuffmanTree(const.StdDCLuminanceLengths, const.StdDCLuminanceValues)

    huffmanDecoding_Y = HuffmanDecoding(yCbCrChannels[0], blockCount, huffmanTreeAcY, huffmanTreeDcY)
    huffmanDecoding_Cb = HuffmanDecoding(yCbCrChannels[1], blockCount, huffmanTreeAcCx, huffmanTreeDcCx)
    huffmanDecoding_Cr = HuffmanDecoding(yCbCrChannels[2], blockCount, huffmanTreeAcCx, huffmanTreeDcCx)
    return [huffmanDecoding_Y.result, huffmanDecoding_Cb.result, huffmanDecoding_Cr.result]


def step10_LengthDecode(debugFlag, yCbCrChannels, blockCount):
    print("Step 10 Length Decode")
    runlengthDecode_Y = ChannelRunlengthDecode(yCbCrChannels[0], blockCount)
    runlengthDecode_Cb = ChannelRunlengthDecode(yCbCrChannels[1], blockCount)
    runlengthDecode_Cr = ChannelRunlengthDecode(yCbCrChannels[2], blockCount)
    return [runlengthDecode_Y, runlengthDecode_Cb, runlengthDecode_Cr]


def step11_InverseZickZack(debugFlag, yCbCrChannels, width, height, sampledSize):
    print("Step 11 Inverse ZickZack")
    zickZackY = ChannelInverseZickZack(yCbCrChannels[0], width, height)
    print("  Inversed Channel Y")
    zickZackCb = ChannelInverseZickZack(yCbCrChannels[1], sampledSize[0], sampledSize[1])
    print("  Inversed Channel Cb")
    zickZackCr = ChannelInverseZickZack(yCbCrChannels[2], sampledSize[0], sampledSize[1])
    print("  Inversed Channel Cr")
    return [zickZackY, zickZackCb, zickZackCr]


def step12_InverseDifferentalEncding(debugFlag, yCbCrChannels):
    print("Step 12 Inverse Differential")
    inverseDifferentalY = ChannelInverseDifferentialEncoding(yCbCrChannels[0])
    inverseDifferentalCb = ChannelInverseDifferentialEncoding(yCbCrChannels[1])
    inverseDifferentalCr = ChannelInverseDifferentialEncoding(yCbCrChannels[2])
    return [inverseDifferentalY, inverseDifferentalCb, inverseDifferentalCr]


def step13_Dequantization(debugFlag, yCbCrChannels):
    print("Step 13 Dequantization")
    dequantizationY = ChannelDequantization(yCbCrChannels[0], const.Q50, "layerY")
    print("  Dequan Channel Y")
    dequantizationCb = ChannelDequantization(yCbCrChannels[1], const.Q50, "layerCb")
    print("  Dequan Channel Cb")
    dequantizationCr = ChannelDequantization(yCbCrChannels[2], const.Q50, "layerCr")
    print("  Dequan Channel Cr")
    return [dequantizationY, dequantizationCb, dequantizationCr]


def step14_Idct(debugFlag, yCbCrChannels):
    print("Step 14 Idct")
    idctY = ChannelIdct(yCbCrChannels[0], "layerY")
    idctCb = ChannelIdct(yCbCrChannels[1], "layerCb")
    idctCr = ChannelIdct(yCbCrChannels[2], "layerCr")
    if debugFlag:
        plt.imshow(idctY)
        plt.xlabel('For Y AFTER')
        plt.set_cmap('gray')
        plt.show()

        plt.imshow(idctCb)
        plt.xlabel('For Cb')
        plt.set_cmap('gray')
        plt.show()

        plt.imshow(idctCr)
        plt.xlabel('For Cr')
        plt.set_cmap('gray')
        plt.show()
    return [idctY, idctCb, idctCr]


def step15_ReverseSubsampling(debugFlag, yCbCrChannels, sampleOverX, sampleOverY):
    print("Step 15 Reverse Subsampling")
    result = np.zeros((len(yCbCrChannels[0]), len(yCbCrChannels[0][0]), 3))
    result[:, :, 0] = np.asarray(yCbCrChannels[0])
    found = False
    if sampleOverX == 1 and sampleOverY == 1:
        ReverseSubsampling_TYPE_4_1_1(yCbCrChannels, result)
        found = True
    if sampleOverX == 2 and sampleOverY == 0:
        ReverseSubsampling_TYPE_4_2_0(yCbCrChannels, result)
        found = True
    if sampleOverX == 2 and sampleOverY == 2:
        ReverseSubsampling_TYPE_4_2_2(yCbCrChannels, result)
        found = True
    if sampleOverX == 4 and sampleOverY == 4:
        ReverseSubsampling_TYPE_4_4_4(yCbCrChannels, result)
        found = True
    if found:
        if debugFlag:
            plt.imshow(result[:, :, 0])
            plt.xlabel('ReverseSubsampling For Y')
            plt.set_cmap('gray')
            plt.show()

            plt.imshow(result[:, :, 1])
            plt.xlabel('ReverseSubsampling For Cb')
            plt.set_cmap('gray')
            plt.show()

            plt.imshow(result[:, :, 2])
            plt.xlabel('ReverseSubsampling For Cr')
            plt.set_cmap('gray')
            plt.show()
        return result
    else:
        raise ValueError("Wrong sample type")


def step16_ConvertYCbCrToRGB(debugFlag, yCbCrChannels, RGBMatrix, yPbPrMatrix):
    print("Step 15 Convert back to RGB")
    height = len(yCbCrChannels)
    width = len(yCbCrChannels[0])
    rgbPicture = np.zeros((width, height, 3))

    # for each YCbCr pixel
    for x in range(width):
        for y in range(height):
            # convert YCbCr to a single vector for calculating
            yCbCrVector = np.asarray([[yCbCrChannels[x, y, 0]], [yCbCrChannels[x, y, 1]], [yCbCrChannels[x, y, 2]]])
            yPbPr = minus(yCbCrVector, yPbPrMatrix)
            rgb = mult(RGBMatrix, yPbPr)
            # map and set RGB value from vector
            rgbPicture[x][y][0] = mapit(rgb[0][0]) / 255.
            rgbPicture[x][y][1] = mapit(rgb[1][0]) / 255.
            rgbPicture[x][y][2] = mapit(rgb[2][0]) / 255.
    return rgbPicture
