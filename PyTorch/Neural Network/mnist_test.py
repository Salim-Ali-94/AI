import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as loader
from torch.utils.data import TensorDataset as group
from sklearn.model_selection import train_test_split as split
import torchvision.datasets as datasets
import neural_network as network


if __name__ == "__main__":

	batch, train_percent = 32, 80
	X, y = network.extract("mnist", "MNIST")
	Y = [element.item() for element in y]
	output = len(list(set(Y)))
	dimension = len(X[0].flatten())
	trainer, tester, validater = network.partition(X, y, output, batch, train_percent)
	episodes = int(1e2)
	learning_rate = 5e-3
	neurons = (dimension, 40, 50, 50, 20, output)
	activity = ("relu", "relu", "relu", "relu", "")
	cost = "crossentropy"
	optimizer = "adam"
	model, error, accuracy = network.learn(trainer, neurons, activity, learning_rate, episodes, optimizer, cost)
	network.plot(error, "forestgreen", "accumulated_error_over_each_epoch_mnist_classification", "Episode", "Error")
	network.plot(accuracy, "mediumvioletred", "accuracy_over_each_epoch_mnist_classification", "Episode", "Learning accuracy")
	network.test(model, tester, output)
