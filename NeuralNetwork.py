import random
import math
import numpy as np


class Neuron:
    def __init__(self, weights=[], ninput=0, isbias=False):
        self.activation = ninput
        self.weights = weights
        self.bias = isbias

    error = 0

    def setinput(self, newactivation):
        self.activation = newactivation

    def getalloutputs(self):
        alloutputs = []
        for weight in self.weights:
            alloutputs.append(weight * self.activation)
        return alloutputs

    def updateweights(self, outputerrors, stepsize):
        for i in range(0, len(outputerrors)):
            self.weights[i] += stepsize * self.activation * outputerrors[i]

        #outputerrors = [En1, En2 ..., Enn]
        #self.weights = [Wself,n1, Wself,n2 ..., Wself,nn]
        #Wself,n1 = Wself,n1 + N * Aself * En1


class NeuronLayer:
    def __init__(self, layersize, nextlayersize):
        self.neurons = []
        for i in range(0, layersize):
            weights = []
            for n in range(0, nextlayersize):
                weights.append(random.randint(1, 10)/10)
            self.neurons.append(Neuron(weights))

    def updateweights(self, errorlist, stepsize):
        for neuron in self.neurons:
            neuron.updateweights(errorlist, stepsize)

    def feedinputs(self, inputlist):
        for n in range(0, len(inputlist)):
            self.neurons[n].setinput(inputlist[n])

    def getalloutputs(self):
        alloutputs = []
        for neuron in self.neurons:
            alloutputs.append(neuron.getalloutputs())
        return alloutputs

    def getarrangedoutputs(self):
        arrangedoutputs = []
        for x in range(0, len(self.neurons[0].weights)):
            arrangedoutputs.append([])
        for neuron in self.neurons:
            outputs = neuron.getalloutputs()
            for i in range(0, len(outputs)):
                arrangedoutputs[i].append(outputs[i])
        return arrangedoutputs

    def getallerrors(self):
        errorlist = []
        for neuron in self.neurons:
            if not neuron.bias:
                errorlist.append(neuron.error)
        return errorlist

    def calculateerrors(self, expectedoutputlist, errorlist):
        if len(expectedoutputlist) != 0:
            for i in range(0, len(self.neurons)):
                neuron = self.neurons[i]
                neuron.error = (1 - math.tanh(neuron.activation)) * (expectedoutputlist[i] - neuron.activation)
        else:
            for neuron in self.neurons:
                if not neuron.bias:
                    neuron.error = (1 - math.tanh(neuron.activation)) * sum(
                        w * e for w, e in zip(neuron.weights, errorlist))

    def getallactivations(self):
        activationlist = []
        for neuron in self.neurons:
            activationlist.append(neuron.activation)
        return activationlist

    def getallweights(self):
        allweights = []
        for neuron in self.neurons:
            allweights.append(neuron.weights)
        return allweights

    def setallweights(self, weightlist):
        for i in range(0, len(self.neurons)):
            for w in range(0, len(weightlist)):
                self.neurons[i].weights[w] = weightlist[i][w]


class GenericNeuralNetwork:
    def __init__(self, neuronlayers):
        self.neuronnetwork = []
        for n in range(0, len(neuronlayers)):
            if n != len(neuronlayers) - 1:
                self.neuronnetwork.append(NeuronLayer(neuronlayers[n], neuronlayers[n+1]))
            else:
                self.neuronnetwork.append(NeuronLayer(neuronlayers[n], 0))

    def setinputsfirstlayer(self, inputlist):
        self.neuronnetwork[0].feedinputs(inputlist)

    def feedforward(self):
        for n in range(1, len(self.neuronnetwork)):
            layerinputs = []
            for outputs in self.neuronnetwork[n-1].getarrangedoutputs():
                layerinputs.append(math.tanh(sum(outputs)))
            self.neuronnetwork[n].feedinputs(layerinputs)

    def addbiastonetwork(self, biasvalue, biasweight):
        for n in range(len(self.neuronnetwork)-1):
            biasweights = []
            for i in self.neuronnetwork[n+1].neurons:
                biasweights.append(biasweight)
            self.neuronnetwork[n].neurons.append(Neuron(biasweights, biasvalue, True))

    def addbiastonetwork2(self):
        pass

    def backpropogation(self, expectedoutputlist, stepsize=0.1):
        nnsize = len(self.neuronnetwork)-1
        for n in range(nnsize, -1, -1):
            if n == nnsize:
                self.neuronnetwork[n].calculateerrors(expectedoutputlist, [])
            else:
                errorlist = self.neuronnetwork[n+1].getallerrors()
                self.neuronnetwork[n].calculateerrors([], errorlist)
                self.neuronnetwork[n].updateweights(errorlist, stepsize)

    def trainnetwork(self, n, trainingdata, expectedoutputlist, stepsize=0.1):
        while n > 0:
            n -= 1
            for i in range(0, len(trainingdata)):
                self.setinputsfirstlayer(trainingdata[i])
                self.feedforward()
                self.backpropogation(expectedoutputlist[i], stepsize)

    def runnetworkonset(self, dataset):
        resultlist = []
        for data in dataset:
            self.setinputsfirstlayer(data)
            self.feedforward()
            resultlist.append(self.neuronnetwork[-1].getallactivations())
        return resultlist

    def printallinternaldata(self):
        for x in range(0, len(self.neuronnetwork)):
            print("neuronlayer " + str(x+1), " has these neurons and weights:\n")
            for n in range(0, len(self.neuronnetwork[x].neurons)):
                if self.neuronnetwork[x].neurons[n].bias:
                    print("bias", end='')
                print("Neuron [" + str(x+1) + ", " + str(n+1) + "] has this activation: " + str(self.neuronnetwork[x].neurons[n].activation), end=' ')
                print(" and this error: " + str(self.neuronnetwork[x].neurons[n].error), end='\n')
                print("And these weights:", end=" ")
                for w in range(0, len(self.neuronnetwork[x].neurons[n].weights)):
                    print("W" + str(w+1) + ": " + str(self.neuronnetwork[x].neurons[n].weights[w]), end=" ")
                print("\n")
            print("--------------------------------------------------------------------\n")





