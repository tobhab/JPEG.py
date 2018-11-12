import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Constants import *
from JPEGHelpers import *

def step1_LoadImage(debugFlag, imagePath):
    img = Image.open(imagePath)
    imageArray = np.array(img) # 640x480x4 array

    if debugFlag:
        print(imageArray.shape)
        plt.imshow(imageArray)
        plt.show()
    return imageArray


def step2_ConvertRGBToYCbCr(debugFlag, rgbChannels):
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
                [rgbChannels[x][y][0]],  # RED-Value of the current Pixel
                [rgbChannels[x][y][1]],  # BLUE-Value of the current Pixel
                [rgbChannels[x][y][2]],  # GREEN-Value of the current Pixel
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


def step3_SubSampleTheCbCrChannel(debugFlag, yCbCrChannels, sampleOverX, sampleOverY):
    height = yCbCrChannels.shape[0]
    width = yCbCrChannels.shape[1]
    channelY = yCbCrChannels[:, :, 0]  # The Y Array of the Picture

    channelCb = [np.zeros(height / sampleOverY)[:], np.zeros(width / sampleOverX)[:]]
    channelCr = [np.zeros(height / sampleOverY)[:], np.zeros(width / sampleOverX)[:]]
    for x in range(0, width, sampleOverX):
        for y in range(0, height, sampleOverY):
            channelCb[y / sampleOverY][x / sampleOverX] = arithmeticMean(yCbCrChannels, x, y, 1, sampleOverX, sampleOverY)
        channelCr[y / sampleOverY][x / sampleOverX] = arithmeticMean(yCbCrChannels, x, y, 2, sampleOverX, sampleOverY)
    return [channelY, channelCb, channelCr]


def step4_DCTAllChannels(debugFlag, yCbCrChannels, blockSize):
    dctY = ChannelDCT(yCbCrChannels[0], blockSize)
    dctCb = ChannelDCT(yCbCrChannels[1], blockSize)
    dctCr = ChannelDCT(yCbCrChannels[2], blockSize)
    return [dctY, dctCb, dctCr]

