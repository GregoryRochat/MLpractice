import IrisNeuralNetwork as inn
import time


def irisdatasetgenerator():
    irisdatalist = []
    trainingsets = []
    validationsets = []
    with open("iris.txt") as data:
        for line in data:
            line = line.strip("\n")
            line = line.split(",")
            irisdatalist.append(line)

    irissetosalist = []
    irisversicolorlist = []
    irisvirginicalist = []

    for iris in irisdatalist:
        if "Iris-setosa" in iris:
            irissetosalist.append([float(iris[0]), float(iris[1]), float(iris[2]), float(iris[3]), iris[4]])
        elif "Iris-versicolor" in iris:
            irisversicolorlist.append([float(iris[0]), float(iris[1]), float(iris[2]), float(iris[3]), iris[4]])
        else:
            irisvirginicalist.append([float(iris[0]), float(iris[1]), float(iris[2]), float(iris[3]), iris[4]])

    for i in range(5, 10, 5):
        trainingsets.append(irissetosalist[:i] + irisversicolorlist[:i] + irisvirginicalist[:i])
        validationsets.append(irissetosalist[i:] + irisversicolorlist[i:] + irisvirginicalist[i:])
        print(i)

    zippedsets = []
    return [trainingsets, validationsets, irisdatalist]


def networklayersizesgenerator():
    threelayernetworks = []
    fourlayernetworks = []
    for i in range(1, 21):
        threelayernetworks.append([4, i, 3])
        for n in range(1, 11):
            fourlayernetworks.append([4, i, n, 3])
    return [threelayernetworks, fourlayernetworks]


def analyzenetworks(networklayersizes, irisdatasets):
    iristrainingsets = irisdatasets[0]
    irisvalidationsets = irisdatasets[1]
    irisdatasettotal = irisdatasets[2]
    file1 = open("ThreeLayerNetworkAnalysisV1.txt", "w")
    file1.close()
    file2 = open("FourLayerNetworkAnalysisV1.txt", "w")
    file2.close()
    file3 = open("BestPerformancesPerLayerV1.txt", "w")
    file3.close()
    for layers in networklayersizes:
        for layer in layers:
            print(layer)
            bestpeformancelist = [0]
            newperformancelist = [0]
            for i in range(0, len(iristrainingsets)):
                for n in range(0, 1):
                    if len(layer) == 3:
                        newperformancelist = analyzenetwork(layer, iristrainingsets[i], irisvalidationsets[i], irisdatasettotal, True)
                    elif len(layer) == 4:
                        newperformancelist = analyzenetwork(layer, iristrainingsets[i], irisvalidationsets[i], irisdatasettotal, False)
                    else:
                        print("Error")
                    if newperformancelist[0] > bestpeformancelist[0]:
                        bestpeformancelist = newperformancelist
            file3 = open("BestPerformancesPerLayerV1.txt", "a")
            file3.write("Network " + str(layer) + ":\n" + "Best performance: " + str(
                bestpeformancelist[0]) + "% on total data set, " + str(
                bestpeformancelist[6]) + "% on validation data, with trainingset size: " + str(
                bestpeformancelist[1]) + "and a bias of: " + str(bestpeformancelist[2]) + ", after training " + str(
                bestpeformancelist[5]) + " times.\n" + "Using these starting weights: " + str(
                bestpeformancelist[3]) + ", and these final weights: " + str(
                bestpeformancelist[4]) + "Time to train was: " + str(
                bestpeformancelist[-1]) + "\n\n-----------------\n")
            file3.close()


def analyzenetwork(layersize, trainingset, validationset, totaldataset, tfile, n=11):
    if tfile:
        file = open("ThreeLayerNetworkAnalysis.txt", "a")
    else:
        file = open("FourLayerNetworkAnalysis.txt", "a")
    c = 1
    # biases = [-3, -1, -0.5]
    biases = [-1]
    bestperformance = 0
    besttrainingsetsize = 0
    beststartweights = []
    bestbias = 0
    bestweights = []
    timestrained = 0
    perftime = 0
    bestperformancev = 0
    file.write("Network " + str(layersize) + ":\n")
    for bias in biases:
        irisnn = inn.IrisNeuralNetwork(layersize)
        irisnn.addbiastonetwork(bias, 1)
        timestart = time.perf_counter()
        while c < n+1:
            irisnn.trainirispredicition(100, trainingset)
            performancev = irisnn.runpredicitions(validationset)
            performancet = irisnn.runpredicitions(totaldataset)

            if performancet > bestperformance:
                bestperformance = performancet
                bestperformancev = performancev
                besttrainingsetsize = len(trainingset)
                beststartweights = irisnn.startingweights
                timestrained = (c * 100)
                bestbias = irisnn.neuronnetwork[0].neurons[-1].activation
                bestweights = []
                perftime = time.perf_counter() - timestart
                for nl in irisnn.neuronnetwork:
                    bestweights.append(nl.getallweights())

            if c == 1 or c == 10 or c == 100:
                file.write("Times trained: " + str(c * 100) + ", trainingset size: " + str(
                    len(trainingset)) + ", Bias: " + str(bias) + "\n" + "Performance on validationset: " + str(
                    performancev) + "%, Performance on totalset: " + str(
                    performancet) + "%\n" + "Total time elapsed: " + str(
                    time.perf_counter() - timestart) + " seconds\n\n---------------------------------\n")
            c += 1
        c = 1
    file.close()
    return [bestperformance, besttrainingsetsize, bestbias, beststartweights, bestweights, timestrained, bestperformancev, perftime]


if __name__ == '__main__':
    performancetime = time.perf_counter()
    networkls = networklayersizesgenerator()
    irisds = irisdatasetgenerator()
    print(networkls)
    print(irisds)
    analyzenetworks(networkls, irisds)


