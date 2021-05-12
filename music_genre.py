from pathlib import Path

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile
import os
import pickle
import random
import operator

import math

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = "model.dat"


# a function to get the distance between feature vectors and find neighbors:
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


# Define a function to get the distance between feature vectors and find neighbors:
def getNeighbors(training_set, instance, k):
    distances = []
    for x in range(len(training_set)):
        dist = distance(training_set[x], instance, k) + distance(instance, training_set[x], k)
        distances.append((training_set[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# Identify the nearest neighbors:
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


# Define a function for model evaluation:
def getAccuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return 1.0 * correct / len(test_set)


#  Extract features from the dataset and dump these features into a binary .dat file “my.dat”:
def extract_features(filename):
    directory = os.path.join(BASE_DIR, 'Data', 'genres_original')
    f = open(filename, 'wb')
    i = 0

    for folder in os.listdir(directory):
        i += 1
        if i == 11:
            break
        for file in os.listdir(os.path.join(directory, folder)):
            print(file)
            (rate, sig) = wav.read(os.path.join(directory, folder, file))
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)

    f.close()


# Train and test split on the dataset:
def loadDataset(filename, split, trSet, teSet):
    dataset = []
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


# Make prediction using KNN and get the accuracy on test data:\
def predict(test_set, training_set):
    print("[!] Making prediction using KNN on test data ....")
    _len = len(test_set)
    predictions = []
    for x in range(_len):
        predictions.append(nearestClass(getNeighbors(training_set, test_set[x], 5)))
    print("[+] Accuracy:")
    accuracy1 = getAccuracy(test_set, predictions)
    print(accuracy1)


if __name__ == '__main__':
    extract_features(DATA_FILE)

    training_set = []
    test_set = []
    loadDataset(DATA_FILE, 0.66, training_set, test_set)
    predict(test_set, training_set)
