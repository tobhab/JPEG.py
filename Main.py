import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from JPEG import *

RGB_image = step1_LoadImage(debugFlag=0, imagePath='RGB.png')
yCbCr_image = step2_ConvertRGBToYCbCr(debugFlag=0, rgbChannels=RGB_image)
sampled_yCbCr_image = step3_SubSampleTheCbCrChannel(debugFlag=0, yCbCrChannels=yCbCr_image, sampleOverX=2, sampleOverY=2)
dct_yCbCr_image = step4_DCTAllChannels(debugFlag=0, yCbCrChannels=sampled_yCbCr_image, blockSize=8)
