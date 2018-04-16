#get and store new training elements
class trainingElement:
    def __init__(self, img):
        self.training_set = [img, img, img, img]
        self.training_set_ready = False
        self.current_element = 0

    def training_set_new(self, img):
        #normally don't need to roll elements. it is faster to use this
        if (self.current_element < 4):
            self.training_set[self.current_element] = img
            self.current_element +=1
            if (self.current_element == 4):
                self.training_set_ready = True
                return self.training_set_ready
        else:
            np.roll(self.training_set, -1)#shift to left by 1
            self.training_set[self.current_element - 1] = img#and update the last one
            return self.training_set_ready

    #renews the state
    def get_training_set_wUpdate(self):
        self.current_element = 0
        self.training_set_ready = False
        return self.training_set

    #keeps the state
    def get_training_set_nUpdate(self):
        return self.training_set


#large ring buffer
class RingBuffer:
    def __init__(self, size):
        self.ringBuffer = [None]*(size+1)
        self.size = size + 1
        self.begin = 0
        self.end = 0

    def append(self, newElement):
        self.ringBuffer[self.end] = newElement
        self.end = (self.end+1) % self.size

        if(self.begin == self.end):
            self.begin = (self.begin + 1) % self.size

    def appendMultiple(self, newElements, size):
        for i in range (0, size):
            self.append(newElements[i])

    #don't delete items from the buffer. just rotate the begin/end indices
    def getitem(self, index):
        return self.ringBuffer[(index + self.begin) % self.size]

    def get_seq_of_items(self, index, length):
        index = (index + self.begin) % self.size
        return self.ringBuffer[index: index + length]

    def length(self):
        if self.end > self.begin:
            return self.end - self.begin
        else:
            return self.size - self.begin + self.end