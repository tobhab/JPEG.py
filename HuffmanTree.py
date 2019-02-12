import BitStreamReader


class HuffmanTree:
    leafs = []
    all = 0
    root = None

    def __init__(self, arr):
        self.all = len(arr)
        leafMap = {}
        # create map of elements with their probability
        for key in arr:
            if key in leafMap:
                leafMap[key].probability += 1
            else:
                leafMap[key] = Node(key)
        self.leafs = leafMap
        # leafs.sort(Comparator.comparingInt(X -> X.probability))
        self.root = self.createTree()

    def __init__(self, lengths, values):
        root = Node()
        current = root
        leafs = []
        levelNumber = 0
        indexInValues = 0
        while True:
            nodesLeftInLevel = lengths[levelNumber]
            levelNumber += 1
            while nodesLeftInLevel > 0:
                test = Node(values[indexInValues])
                indexInValues += 1
                leafs.add(test)
                nodesLeftInLevel -= 1
                current = current.addLeftMost(test)
            current = current.traverseDown()
            if levelNumber >= len(lengths):
                break

    def createTree(self):
        tmp = []
        while len(self.leafs) > 1:
            while len(self.leafs) > 1:
                leftChild = self.leafs.pop(0)
                rightChild = self.leafs.pop(0)
                newNode = Node(leftChild, rightChild)
                leftChild.parent = newNode
                rightChild.parent = newNode
                root = newNode
                tmp.append(newNode)
            if len(leafs) == 1:
                n = leafs.pop(0)
                tmp.append(n)
            leafs = tmp
        return root

    def getInOrder(self, root):
        returnList = []
        if root.left is None:
            returnList = self.getInOrder(root.left)
        if returnList is None:
            returnList = []
        returnList.append(root)
        if root.right is None:
            returnList.extend(self.getInOrder(root.right))
        return returnList

    def lookUpCodeNumber(self, reader : BitStreamReader):
      return self.root.findValueByCode(reader)


    def writeCodeToWriter(self, writer, codeToWrite):
        for leaf in self.leafs:
            if leaf.value == codeToWrite:
                codeWordToWrite = leaf.getCode()
                self.writeToWriter(writer, codeWordToWrite)
                break

    def writeToWriter(self, writer, codeWord):
        writer.write(codeWord.code, codeWord.bitCount)


# holds the bits right aligned and MSB on the left
class CodeWord:
    code = 0
    bitCount = 0


# Node class represent an element with its value and the probability
class Node:
    isNode = False
    value = 0
    probability: int = 0

    left = None
    right = None
    parent = None

    def __init__(self, value):
        self.value = value
        self.probability = 1
        self.isNode = False

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.isNode = True
        if left is not None and right is not None:
            self.probability = left.probability + right.probability

    def __init__(self, parent):
        self.left = None
        self.right = None
        self.isNode = True
        self.parent = parent

    def getCodeAsString(self):
        if self.parent is None:
            return ""
        return self.parent.getCodeAsString(self)

    def getCodeAsString(self, caller):
        if caller == self.left:
            return self.getCodeAsString() + "0"
        else:
            return self.getCodeAsString() + "1"

    def getCode(self):
        codeWord = self.getCode(CodeWord())
        return codeWord

    def getCode(self, codeWord):
        if self.parent is None:
            return codeWord
        return self.parent.getCode(self, codeWord)

    def getCode(self, caller, codeWord):
        if caller != self.left:
            # rightmost bit must be set to one
            codeWord.code |= 1 << codeWord.bitCount
        codeWord.bitCount += 1
        return self.getCode(codeWord)

    def getDepth(self):
        if self.parent is None:
            return -1
        else:
            return self.parent.getDepth() + 1

    # Add the given node to the left most postition below this node.
    # It tries to traverse the tree on the same level to find a location to add the node
    # @param toAdd
    # @return The parent to which the node was actually added
    def addLeftMost(self, toAdd):
        if self.left is None:
            left = toAdd
            left.parent = self
            return self
        if self.right is None:
            right = toAdd
            right.parent = self
            return self.traverseRight()
        toReturn = self.traverseRight()
        toReturn.addLeftMost(toAdd)
        return toReturn

    # Searches and returns the node which is to the right on the same level in the tree
    # @return The node which is to the right of this node
    def traverseRight(self):
        stepsUp = 0
        current = self
        while current.right is not None:
            if current.parent is None:
                # using runtime exceptions since they don't need to be added to the method signature
                raise ValueError("traverseRight failed -> No parent")
            current = current.parent
            stepsUp += 1
        current.right = Node(current)
        current = current.right
        while --stepsUp > 0:
            current.left = Node(current)
            current = current.left
        return current

    # @return The node which was able to be placed directly below or to the right below this node
    def traverseDown(self):
        if self.left is None:
            self.left = Node(self)
            return self.left
        if self.right is None:
            self.right = Node(self)
            return self.right
        return self.traverseRight().traverseDown()

    def findValueByCode(self, reader : BitStreamReader):
        if not self.isNode:
            return self.value

        nextBit = reader.readBit()
        if nextBit:
            return self.right.findValueByCode(reader)
        else:
            return self.left.findValueByCode(reader)

    def toString(self):
        s = ""
        if self.isNode:
            s += "Node["
        else:
            s += "Leaf["
        s += "Value: " + self.value
        s += " Probability: " + self.probability
        s += "]\n"
        return s
