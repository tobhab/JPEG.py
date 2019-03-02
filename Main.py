import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import Constants as consts
from JPEG import *

sampling = [2, 0]
# sampling = [2, 2]
# sampling = [4, 4]
# sampling = [2, 0]

RGB_image = step1_LoadImage(debugFlag=0, imagePath='RGB.png')
image_width, image_height = RGB_image.shape[0], RGB_image.shape[1]
blockCount = (image_width / consts.BlockSize) * (image_height / consts.BlockSize)
yCbCr_image = step2_ConvertRGBToYCbCr(debugFlag=0, rgbChannels=RGB_image)
sampled_yCbCr_image = step3_SubSample(debugFlag=0, yCbCrChannels=yCbCr_image, sampleOverX=sampling[0], sampleOverY=sampling[1])
sampled_size = sampled_yCbCr_image[1].shape
dct_yCbCr_image = step4_DCTAllChannels(debugFlag=0, yCbCrChannels=sampled_yCbCr_image)
quantization_yCbCr_image = step5_Quantization(debugFlag=0, yCbCrChannels=dct_yCbCr_image)
differental_yCbCr_image = step5_DifferentalEncoding(debugFlag=0, yCbCrChannels=quantization_yCbCr_image)
zickzack_yCbCr_image = step6_ZickZack(debugFlag=0, yCbCrChannels=differental_yCbCr_image)
# length_encode_yCbCr_image = step7_LengthEncode(debugFlag=0, yCbCrChannels=zickzack_yCbCr_image)
# huffman_encode_yCbCr_image = step8_HuffmanEncode(debugFlag=0, yCbCrChannels=length_encode_yCbCr_image, zicks=zickzack_yCbCr_image)
#
# huffman_decode_yCbCr_image = step9_HuffmanDecode(debugFlag=0, yCbCrChannels=huffman_encode_yCbCr_image, blockCount=blockCount)
# length_decode_yCbCr_image = step10_LengthDecode(debugFlag=0, yCbCrChannels=length_encode_yCbCr_image, blockCount=blockCount)
inverse_zickzack_image = step11_InverseZickZack(debugFlag=0, yCbCrChannels=zickzack_yCbCr_image, width=image_width, height=image_height, sampledSize=sampled_size)
inverse_differental_yCbCr_image = step12_InverseDifferentalEncding(debugFlag=0, yCbCrChannels=inverse_zickzack_image)
dequantization_yCbCr_image = step13_Dequantization(debugFlag=0, yCbCrChannels=inverse_differental_yCbCr_image)
idct_yCbCr_image = step14_Idct(debugFlag=0, yCbCrChannels=dequantization_yCbCr_image)
reversed_subsampling_yCbCr_image = step15_ReverseSubsampling(debugFlag=0, yCbCrChannels=idct_yCbCr_image, sampleOverX=sampling[0], sampleOverY=sampling[1])
converted_RGB_image = step16_ConvertYCbCrToRGB(debugFlag=0, yCbCrChannels=reversed_subsampling_yCbCr_image, RGBMatrix=const.RGBMatrix1, yPbPrMatrix=const.YCbCrMatrix)

# cache = np.ones([64, 64, 4], dtype=int)
# cache[:, :, :] = 255
# cache[:, :, 0:3] = (converted_RGB_image * 255.0).astype(int)

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(RGB_image)
axarr[0].set_title("Before")
axarr[1].imshow(converted_RGB_image)
axarr[1].set_title("After")
plt.show()

