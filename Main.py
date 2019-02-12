import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import Constants as consts
from JPEG import *

RGB_image = step1_LoadImage(debugFlag=0, imagePath='RGB2.png')
image_width, image_height = RGB_image.shape[0], RGB_image.shape[1]
blockCount = (image_width / consts.BlockSize) * (image_height / consts.BlockSize)
yCbCr_image = step2_ConvertRGBToYCbCr(debugFlag=0, rgbChannels=RGB_image)
sampled_yCbCr_image = step3_SubSample(debugFlag=0, yCbCrChannels=yCbCr_image, sampleOverX=2, sampleOverY=2)
dct_yCbCr_image = step4_DCTAllChannels(debugFlag=0, yCbCrChannels=sampled_yCbCr_image)
quantization_yCbCr_image = step5_Quantization(debugFlag=0, yCbCrChannels=dct_yCbCr_image)
differental_yCbCr_image = step5_DifferentalEncoding(debugFlag=0, yCbCrChannels=quantization_yCbCr_image)
zickzack_yCbCr_image = step6_ZickZack(debugFlag=0, yCbCrChannels=differental_yCbCr_image)
# length_encode_yCbCr_image = step7_LengthEncode(debugFlag=0, yCbCrChannels=zickzack_yCbCr_image)
# huffman_encode_yCbCr_image = step8_HuffmanEncode(debugFlag=0, yCbCrChannels=length_encode_yCbCr_image, zicks=zickzack_yCbCr_image)
#
# huffman_decode_yCbCr_image = step9_HuffmanDecode(debugFlag=0, yCbCrChannels=huffman_encode_yCbCr_image, blockCount=blockCount)
# length_decode_yCbCr_image = step10_LengthDecode(debugFlag=0, yCbCrChannels=length_encode_yCbCr_image, blockCount=blockCount)
inverse_zickzack_image = step11_InverseZickZack(debugFlag=0, yCbCrChannels=zickzack_yCbCr_image, width=image_width, height=image_height)
inverse_differental_yCbCr_image = step12_InverseDifferentalEncding(debugFlag=0, yCbCrChannels=inverse_zickzack_image)
dequantization_yCbCr_image = step13_Dequantization(debugFlag=0, yCbCrChannels=inverse_differental_yCbCr_image)
idct_yCbCr_image = step14_Idct(debugFlag=0, yCbCrChannels=dequantization_yCbCr_image)
reversed_subsampling_yCbCr_image = step15_ReverseSubsampling(debugFlag=0, yCbCrChannels=idct_yCbCr_image, sampleOverX=2, sampleOverY=0)
converted_RGB_image = step16_ConvertYCbCrToRGB(debugFlag=0, yCbCrChannels=reversed_subsampling_yCbCr_image, RGBMatrix=const.RGBMatrix1, yPbPrMatrix=const.YCbCrMatrix)

fig=plt.figure(figsize=(1, 2))
imgs = [converted_RGB_image, RGB_image]
for i in range(1, len(imgs) + 1):
    img = converted_RGB_image
    fig.add_subplot(1, 2, i)
    plt.imshow(img)
plt.show()

