import NeuralNetwork as nn


class IrisNeuralNetwork(nn.GenericNeuralNetwork):
    def __init__(self, neuronlayers):
        super().__init__(neuronlayers)
        self.startingweights = []
        for n in self.neuronnetwork:
            self.startingweights.append(n.getallweights())

    def trainirispredicition(self, n, irisdataset, stepsize=0.1):
        inputdataset = []
        expectedoutputs = []
        for iris in irisdataset:
            inputdataset.append([float(iris[0]), float(iris[1]), float(iris[2]), float(iris[3])])
            if iris[4] == "Iris-setosa":
                expectedoutputs.append([1, 0, 0])
            elif iris[4] == "Iris-versicolor":
                expectedoutputs.append([0, 1, 0])
            else:
                expectedoutputs.append([0, 0, 1])
        self.setinputsfirstlayer(inputdataset[0])
        self.trainnetwork(n, inputdataset, expectedoutputs, stepsize)

    def runpredicitions(self, irisdataset):
        inputdataset = []
        expectedoutputs = []
        actualoutputs = []
        predictedoutputs = []
        numcorrect = 0
        for iris in irisdataset:
            inputdataset.append([float(iris[0]), float(iris[1]), float(iris[2]), float(iris[3])])
            if iris[4] == "Iris-setosa":
                expectedoutputs.append([1, 0, 0])
            elif iris[4] == "Iris-versicolor":
                expectedoutputs.append([0, 1, 0])
            else:
                expectedoutputs.append([0, 0, 1])
            actualoutputs.append(iris[4])
        results = self.runnetworkonset(inputdataset)
        for result in results:
            activeneuron = result.index(max(result))
            if activeneuron == 0:
                predictedoutputs.append("Iris-setosa")
            elif activeneuron == 1:
                predictedoutputs.append("Iris-versicolor")
            else:
                predictedoutputs.append("Iris-virginica")
        for i in range(0, len(predictedoutputs)):
            # print("Predicted: ", predictedoutputs[i], " - Actual: ", actualoutputs[i])
            if predictedoutputs[i] == actualoutputs[i]:
                numcorrect += 1
        # print("Percentage correct: ", numcorrect / len(actualoutputs) * 100)
        return numcorrect / len(actualoutputs) * 100







# irisnn = IrisNeuralNetwork([4, 5, 4, 3])
# irisnn.addbiastonetwork(-1, 1)
# for x in range(0, 100):
    # irisnn.trainirispredicition(1000, trainingset)
    # irisnn.runpredicitions(my_list)
# print("\n")