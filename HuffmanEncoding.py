import math

import RunlengthEncode
from BitStreamWriter import BitStreamWriter


class HuffmanEncoding:
    result = []

    def __init__(self, arr, encodingTreeAC, encodingTreeDC):
        out = []
        writer = BitStreamWriter(out)
        nextIndex = 0

        # Encode all blocks for the entire array
        while nextIndex < arr.length:
            # Encode DC component
            dcValue = arr[nextIndex]
            nextIndex += 1
            dcValueBitWidth = self.getBitWidth(dcValue)
            encodingTreeDC.writeCodeToWriter(writer, dcValueBitWidth)
            if dcValueBitWidth == 0:  # special case of a DC block being a EOB marker, keep going with the next block
                continue
            if dcValue <= 0:
                dcValue = int(2 ** dcValueBitWidth) - 1 + dcValue
            writer.write(dcValue, dcValueBitWidth)
            # Encode ACs till an EOB is reached
            while nextIndex < arr.length and arr[nextIndex] != RunlengthEncode.endOfBlockMarker:
                runlength = arr[nextIndex]
                nextIndex += 1
                if runlength == RunlengthEncode.longZeroRunMarker:
                    # Encode LZR
                    encodingTreeAC.writeCodeToWriter(writer, 0xF0)
                    continue  # keep going with the next value in this block
                acValue = arr[nextIndex]
                nextIndex += 1
                bitsize = self.getBitWidth(acValue)
                if acValue <= 0:
                    acValue = int(2 ** bitsize) - 1 + acValue
                # Encode (runlength,bitsize)(value)
                encodingTreeAC.writeCodeToWriter(writer, (runlength << 4) | bitsize)
                writer.write(acValue, bitsize)
            nextIndex  # We need to skip the EOB marker
            nextIndex += 1
            # Encode EOB
            encodingTreeAC.writeCodeToWriter(writer, 0x00)
        writer.close()
        self.result = out.toByteArray()

    def getBitWidth(self, value):
        value = math.abs(value)
        returnValue = math.ceil(self.log(value + 1, 2))
        return int(returnValue)

    @staticmethod
    def log(x, base):
        return math.log(x) / math.log(base)

    def getResult(self):
        return self.result
