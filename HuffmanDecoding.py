import numpy as np

import HuffmanTree
import RunlengthEncode
from BitStreamReader import BitStreamReader


class HuffmanDecoding:
    result = []

    def __init__(self, arr, blockCount, blockWidth, decodingTreeAC: HuffmanTree, decodingTreeDC: HuffmanTree):
        reader = BitStreamReader(arr)

        # create an array which could hold the largest possible resulting data and cut off the unneeded parts later
        self.result = np.zeros(blockCount * (blockWidth * blockWidth));
        indexInResult = 0

        while blockCount > 0:
            blockCount -= 1
            # read dc value code
            dcValueCode = decodingTreeDC.lookUpCodeNumber(reader)
            # skip to next block if it's a EOB marker
            if dcValueCode == 0:
                self.result[indexInResult] = 0
                indexInResult += 1
                continue
            # read dc value
            dcValue = reader.readBits(dcValueCode)
            if dcValue < 2 ** dcValueCode - 1:
                twoPowN = int(2 ** dcValueCode)
                dcValue = dcValue - twoPowN + 1
                self.result[indexInResult] = dcValue
            indexInResult += 1

            # read ac values till a EOB is reached
            while True:
                acValueCode = decodingTreeAC.lookUpCodeNumber(reader)
                if acValueCode == 0x00:  # EOB
                    self.result[indexInResult] = RunlengthEncode.endOfBlockMarker
                    indexInResult += 1
                    break
                else:
                    if acValueCode == 0xF0:  # LZR
                        self.result[indexInResult] = RunlengthEncode.longZeroRunMarker
                        indexInResult += 1
                        continue
                    else:
                        acRunLength = acValueCode >> 4
                        self.result[indexInResult] = acRunLength
                        indexInResult += 1
                        acBitLength = acValueCode & 0xF
                        acValue = reader.readBits(acBitLength)
                        if acValue < 2 ** acBitLength - 1:
                            twoPowN = int(2 ** acBitLength)
                            acValue = acValue - twoPowN + 1
                        self.result[indexInResult] = acValue
                        indexInResult += 1

        self.result = self.result[0:indexInResult]

    def getResult(self):
        return self.result
