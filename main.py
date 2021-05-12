import sys

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import pickle
import operator
from collections import defaultdict
import numpy as np
from music_genre import DATA_FILE


def loadDataset(filename):
    dataset = []
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    return dataset


# loadDataset("my.dat")


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


def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


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


# i = 1
# for folder in os.listdir("./Data/genres_original"):
#     results[i] = folder
#     i += 1


def predict(file_name, dataset):
    results = defaultdict(int)
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    i = 1
    for genre in genres:
        results[i] = genre
        i += 1
    # (rate, sig) = wav.read("./Data/genres_original/disco/disco.00005.wav")
    (rate, sig) = wav.read(file_name)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, 0)

    pred = nearestClass(getNeighbors(dataset, feature, 5))

    print(file_name + " : " + results[pred])


if __name__ == '__main__':
    DATASET_FILE = ""
    input_files = sys.argv[1:]
    dataset = DATA_FILE
    _dataset = loadDataset(dataset)
    for input_file in input_files:
        predict(file_name=input_file, dataset=_dataset)
