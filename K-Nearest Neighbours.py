import numpy as np
import math


def getDistance(point, point2):
    """Get's the distance between 2 point"""
    result = 0
    for i in range(0, len(point)):
        result += (point[i] - point2[i]) ** 2
    return math.sqrt(result)


def getLabelList(dates):
    labels = []
    for label in dates:
        label = label % 10000
        if label < 301:
            labels.append('winter')
        elif 301 <= label < 601:
            labels.append('lente')
        elif 601 <= label < 901:
            labels.append('zomer')
        elif 901 <= label < 1201:
            labels.append('herfst')
        else:  # from 01-12 to end of year
            labels.append('winter')
    return labels


class DistanceWithLabel:
    def __init__(self, distance, label):
        self.distance = distance
        self.label = label

    def getLabel(self):
        return self.label

    def getDistance(self):
        return self.distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __repr__(self):
        return "Label: " + str(self.label) + " Distance: " + str(self.distance)


class KNearest:
    def __init__(self, dataset, labels):
        self.k = 1
        self.data = dataset
        self.labels = labels

    def setK(self, k):
        if k < len(self.data) and k >= 2:
            self.k = k
        return self.k

    def findOptimalK(self, validationset, validationsetLabels):
        numCorrect = 0
        optimalK = [0, 0]
        size = len(validationset)
        file = open("temp2.txt", "w")
        file.close()
        file = open("temp2.txt", "a")
        while self.k < size:
            for i in range(0, size):
                label = self.run(validationset[i],file)
                if label == validationsetLabels[i]:
                    numCorrect += 1
            numCorrect = numCorrect * 100 / size
            print(self.k, numCorrect)
            if numCorrect > optimalK[1]:
                optimalK = [self.k, numCorrect]
            self.setK(self.k + 1)
            numCorrect = 0
        self.setK(optimalK[0])
        file.close()
        return optimalK

    def run(self, point, file):
        distanceWithLabelList = []
        for i in range(0, self.k):
            gd = getDistance(point, self.data[i])
            distanceWithLabelList.append(DistanceWithLabel(
                gd, self.labels[i]))
        distanceWithLabelList.sort(reverse=True)
        for i in range(self.k, len(self.data)):
            pointDistance = getDistance(point, self.data[i])
            if pointDistance < distanceWithLabelList[0].getDistance():
                distanceWithLabelList[0] = DistanceWithLabel(
                    pointDistance, self.labels[i])
                distanceWithLabelList.sort(reverse=True)
        possibleLabels = {}
        file.write("k: " + str(self.k))
        for distanceWithLabel in distanceWithLabelList:
            file.write(" (" + str(distanceWithLabel.getDistance()) + " - " + str(distanceWithLabel.getLabel()) + ")")
            if distanceWithLabel.getLabel() in possibleLabels:
                possibleLabels[distanceWithLabel.getLabel()] += 1
            else:
                possibleLabels[distanceWithLabel.getLabel()] = 1
        resultList = sorted(possibleLabels.items(),
                            key=lambda x: x[1], reverse=True)
        file.write("\n")
        return resultList[0][0]


def main():
    dataset = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7], converters={
                            5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    labels = np.genfromtxt('dataset1.csv',
                           delimiter=';', usecols=[0])
    validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7], converters={
                               5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    datesValidation = np.genfromtxt(
        'validation1.csv', delimiter=';', usecols=[0])
    kNearest = KNearest(dataset, getLabelList(labels))
    print(kNearest.findOptimalK(validation, getLabelList(datesValidation)))


if __name__ == '__main__':
    main()

