import numpy as np

# STATIC STUFF
BlockSize = 8
BlockLength = BlockSize**2
EndOfBlockMarker = -1
LongZeroRunMarker = 17

MultiplicationMatrixYCbCr = np.asarray([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
])

AdditionMatrixYCbCr =np.asarray( [
    [0.0],
    [128.0],
    [128.0]
])

ZickZackMappingForBlockSize8 =np.asarray( [
    [1, 2, 6, 7, 15, 16, 28, 29],
    [3, 5, 8, 14, 17, 27, 30, 43],
    [4, 9, 13, 18, 26, 31, 42, 44],
    [10, 12, 19, 25, 32, 41, 45, 54],
    [11, 20, 24, 33, 40, 46, 53, 55],
    [21, 23, 34, 39, 47, 52, 56, 61],
    [22, 35, 38, 48, 51, 57, 60, 62],
    [36, 37, 49, 50, 58, 59, 63, 64]])

ReverseZickZackMappingForBlockSize8 = np.asarray( [
    [0, 0], [0, 1], [1, 0], [2, 0], [1, 1], [0, 2], [0, 3], [1, 2],
    [2, 1], [3, 0], [4, 0], [3, 1], [2, 2], [1, 3], [0, 4], [0, 5],
    [1, 4], [2, 3], [3, 2], [4, 1], [5, 0], [6, 0], [5, 1], [4, 2],
    [3, 3], [2, 4], [1, 5], [0, 6], [0, 7], [1, 6], [2, 5], [3, 4],
    [4, 3], [5, 2], [6, 1], [7, 0], [7, 1], [6, 2], [5, 3], [4, 4],
    [3, 5], [2, 6], [1, 7], [2, 7], [3, 6], [4, 5], [5, 4], [6, 3],
    [7, 2], [7, 3], [6, 4], [5, 5], [4, 6], [3, 7], [4, 7], [5, 6],
    [6, 5], [7, 4], [7, 5], [6, 6], [5, 7], [6, 7], [7, 6], [7, 7]])


def ZickZackMappingForBlocksize(blockSize):
    indexArray = None
    if blockSize == 8:
        indexArray = ZickZackMappingForBlockSize8
    # Add new Mappings for Blocksizes here!
    # if blockSize = ?:
    #     indexArray = consts.ZickZackMappingForBlocksize?
    if indexArray is None:
        raise Exception("No Index Array for given Blocksize")
    return indexArray


def ReverseZickZackMappingForBlocksize():
    indexArray = None
    if BlockSize == 8:
        indexArray = ReverseZickZackMappingForBlockSize8
    # Add new Mappings for Blocksizes here!
    # if BlockSize = ?:
    #     indexArray = consts.ZickZackMappingForBlocksize?
    if indexArray is None:
        raise Exception("No Index Array for given Blocksize")
    return indexArray


# The data for the publically defined tables, as specified in ITU T.81
# JPEG specification section K3.3 and used in the IJG library.
StdDCLuminanceLengths = np.asarray( [
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
])

StdDCLuminanceValues = np.asarray([
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b,
])

StdDCChrominanceLengths =np.asarray( [
    0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
])

StdDCChrominanceValues =np.asarray( [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b,
])

StdACLuminanceLengths = np.asarray([
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03,
    0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d,
])

StdACLuminanceValues =np.asarray( [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
])

StdACChrominanceLengths =np.asarray( [
    0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04,
    0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77,
])

StdACChrominanceValues = np.asarray([
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
])

Q20 =np.asarray( [
    [3, 5, 7, 9, 11, 13, 15, 17],
    [5, 7, 9, 11, 13, 15, 17, 19],
    [7, 9, 11, 13, 15, 17, 19, 21],
    [9, 11, 13, 15, 17, 19, 21, 23],
    [11, 13, 15, 17, 19, 21, 23, 25],
    [13, 15, 17, 19, 21, 23, 25, 27],
    [15, 17, 19, 21, 23, 25, 27, 29],
    [17, 19, 21, 23, 25, 27, 29, 31]
])

Q50 = np.asarray( [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q70 =np.asarray( [
    [10, 15, 25, 37, 51, 66, 82, 100],
    [15, 19, 28, 39, 52, 67, 83, 101],
    [25, 28, 35, 45, 58, 72, 88, 105],
    [37, 39, 45, 54, 66, 79, 94, 111],
    [51, 52, 58, 66, 76, 89, 103, 119],
    [66, 67, 72, 79, 89, 101, 114, 130],
    [82, 83, 88, 94, 103, 114, 127, 142],
    [100, 101, 105, 111, 119, 130, 142, 156]
])

Q100 = np.asarray([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

Q255 = np.asarray( [
    [80, 60, 50, 80, 120, 200, 255, 255],
    [55, 60, 70, 95, 130, 255, 255, 255],
    [70, 65, 80, 120, 200, 255, 255, 255],
    [70, 85, 110, 145, 255, 255, 255, 255],
    [90, 110, 185, 255, 255, 255, 255, 255],
    [120, 175, 255, 255, 255, 255, 255, 255],
    [245, 255, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255, 255]
])

RGBMatrix1 = np.asarray([
    [1.0, 0, 1.402],
    [1.0, -0.344136, -0.714136],
    [1.0, 1.772, 0]
])

YCbCrMatrix = np.asarray( [
    [0.0],
    [128.0],
    [128.0]
])
