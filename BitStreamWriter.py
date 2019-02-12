class BitStreamWriter:
    out = []
    #  Using a long, since when there are 7 bits already in the buffer,
    #  then the user is still able to add an entire 32bit and that would lead to data loss.
    #  The newest bit is always added to the right.
    bits = 0
    storedBits = 0
    # Only use the 8 right most bits, since that is the definition in OutputStream
    bitsStoredAtOnce = 8

    def __init__(self, out):
        out = out

    def writeBoolean(self, value):
        self.bits <<= 1  # Make room on the right side to ...
        self.bits |= 1 if value else 0  # ...push in another bit there.
        self.storedBits += 1
        self.writeFullBytes()

    def writeInt(self, values):
        self.write(values, 32)

    # Writes the n rightmost bits into the stream in the order MSB to LSB
    def writeRightBits(self, values, n):
        andBy = ((1 << n) - 1)  # Generate a mask for the lower bits, to...
        values = values & andBy  # ...zero the unused top bits to avoid data corruption
        self.bits <<= n  # Make room on the right side to ...
        self.bits |= values  # ...push new values in there
        self.storedBits += n
        self.writeFullBytes()

    def writeFullBytes(self):
        while self.storedBits >= self.bitsStoredAtOnce:
            self.storedBits -= self.bitsStoredAtOnce
            outputBits = int((self.bits >> self.storedBits))
            self.out.write(outputBits)

    def __del__(self):
        if self.storedBits != 0:
            # save the remaining bits and do padding with 1 bits according to spec
            self.write(-1, self.bitsStoredAtOnce - (self.storedBits % self.bitsStoredAtOnce))
        self.out.close()
