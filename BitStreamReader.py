class BitStreamReader:
    input = []
    # the next bit to read is at the left most position in self long
    bits = 0
    storedBits = 0
    index = 0

    def __init__(self, input):
        self.input = input

    def readBit(self):
        return self.readBits(1) == 1

    def readBits(self, count):
        self.ensureBuffer(count)
        self.storedBits -= count
        shiftBy = 64 - count
        bitsToReturn = int(self.bits >> shiftBy)
        self.bits <<= count
        return bitsToReturn

    def readInt(self):
        return self.readBits(32)

    def readNextInput(self):
        if self.index < len(input):
            resu = input[self.index] & 0xff
            self.index += 1
            return resu
        return -1

    def ensureBuffer(self, count):
        while self.storedBits < count:
            nextBits = self.readNextInput()  # Discard all but the 8 rightmost bits, because only the last byte holds actual data
            if nextBits == -1:
                raise Exception("Not enough bits left in stream")
            self.storedBits += 8
            self.bits |= nextBits << (64 - self.storedBits)



    # def __del__(self):
    #   self.input.close()
