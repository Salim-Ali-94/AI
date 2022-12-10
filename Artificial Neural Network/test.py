import numpy as np
import neural_network as network
import pandas as pd


def initialize():

	data = pd.read_csv("iris.csv")
	labels = data.iloc[0:100, 4].values
	labels = np.where(labels == "setosa", 0, 1)
	remainder = data.iloc[100:150, 4].values
	labels = np.concatenate((labels, remainder))
	labels = np.where(labels == "virginica", 0.5, labels)
	labels = np.copy(labels[np.newaxis].T)
	features = data.iloc[0:150, 0:4].values
	return features, labels

def learn(inputs, outputs, hyper_parameters, learning_rate, episodes, training_data_percent):

	ANN = network.ArtificialNeuralNetwork(inputs, outputs, hyper_parameters, learning_rate, episodes)
	ANN.partition(training_data_percent, 0)
	ANN.train()
	ANN.test()
	ANN.plot()

	
if __name__ == "__main__":

	episodes, learning_rate = 10000, 0.05
	training_data_percent = 75
	x, y = initialize()
	hyper_parameters = (x.shape[1], 40, 20, 10, y.shape[1])
	learn(x, y, hyper_parameters, learning_rate, episodes, training_data_percent)
