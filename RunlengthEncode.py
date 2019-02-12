import numpy as np


class RunlengthEncode:
    result = []
    endOfBlockMarker = -1
    longZeroRunMarker = 17

    # Generates a simplified RLE encoding so that no bit operations are needed to read the values in the resulting array.
    # Each AC value is split up into 2 values: Runlength and Value
    # If a run of 16 zeros is reached the longZeroRunMarker is inserted
    # If the end of block is reached without any more non-zero values then the endOfBlockMarker is inserted
    # @param arr
    # @param block_size
    def __init__(self, arr, block_size):
        # blockLength is the length of an 2d square with the side length of block_size
        blockLength = block_size  # block_size
        # Points to the next free index where data can be written to
        currentResultIndex = 0
        currentBlockOffset = 0
        # This is the longest a RLE could be, we just made sure that this always fits into the destination array.
        # We only return a part of that array after we are done here.
        self.result = np.zeros(arr.length * 2)
        while True:
            currentResultIndex = self.encodeBlock(arr, blockLength, currentBlockOffset, currentResultIndex)
            currentBlockOffset += blockLength
            if currentBlockOffset >= arr.length:
                break
        self.result = self.result[0:currentResultIndex]


    def encodeBlock(self, arr, blockLength, currentBlockOffset, currentResultIndex):
        # Just copy the dc component to the result array, will be compressed in the huffman stage
        self.result[currentResultIndex] = arr[currentBlockOffset]
        currentResultIndex += 1
        indexInCurrentBlock = 0
        while indexInCurrentBlock < (blockLength - 1):
            runLength = 0
            while indexInCurrentBlock + 1 <= (blockLength - 1) and arr[
                currentBlockOffset + indexInCurrentBlock + 1] == 0:
                indexInCurrentBlock += 1
                runLength += 1
                if runLength == 16:
                    # System.out.println("long run")
                    runLength = 0
                    self.result[currentResultIndex] = self.longZeroRunMarker
                    currentResultIndex += 1
            indexInCurrentBlock += 1 # Because of ++X statement
            if indexInCurrentBlock >= blockLength:
                while self.result[currentResultIndex - 1] == self.longZeroRunMarker:
                    # System.out.println("rewind long run")
                    currentResultIndex -= 1
                # System.out.println("EOB @" + (currentBlockOffset + indexInCurrentBlock))
                self.result[currentResultIndex] = self.endOfBlockMarker
                currentResultIndex += 1
            else:
                # System.out.println("(" + runLength + "," + arr[currentBlockOffset + indexInCurrentBlock ] + ")")
                self.result[currentResultIndex] = runLength
                currentResultIndex += 1
                self.result[currentResultIndex] = arr[currentBlockOffset + indexInCurrentBlock]
                currentResultIndex += 1
        # Must make sure that it ends with an end of block marker, if there is a checkerboard pattern then the very last coefficient might not be zero
        if self.result[currentResultIndex - 1] != self.endOfBlockMarker:
            self.result[currentResultIndex] = self.endOfBlockMarker
            currentResultIndex += 1
        return currentResultIndex


    def getResult(self):
        return self.result
