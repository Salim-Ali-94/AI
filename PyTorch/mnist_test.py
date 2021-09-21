import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as loader
from torch.utils.data import TensorDataset as group
from sklearn.model_selection import train_test_split as split
import torchvision.datasets as datasets
import neural_network as network


if __name__ == "__main__":

	X, Y = [], []
	train_percent, validate_percent = 80, 0
	data = datasets.MNIST(root = "dataset/", transform = transforms.ToTensor(), download = True)
	Z = loader(dataset = data, batch_size = 1)
	for x, y in Z: X.append(x), Y.append(y)
	X, Y = torch.cat(X, dim = 0), torch.cat(Y, dim = 0)
	X = X / torch.max(X)
	train_x, test_x, train_y, test_y = split(X, Y, test_size = 1 - (train_percent / 100))
	train = group(train_x, train_y)
	test = group(test_x, test_y)
	flag, batch = None, 32
	trainer = loader(dataset = train, batch_size = batch, shuffle = True)
	tester = loader(dataset = test, batch_size = batch, shuffle = True)
	episodes, output = int(1e2), 10
	learning_rate = 5e-3
	dimension = len(X[0].flatten())
	neurons = (dimension, 40, 50, 50, 20, output)
	activity = ("relu", "relu", "relu", "relu", "")
	cost = "crossentropy"
	optimizer = "adam"
	model, error, accuracy = network.learn(trainer, neurons, activity, learning_rate, episodes, optimizer, cost)
	network.plot(error, "forestgreen", "accumulated_error_over_each_epoch_mnist_classification", "Episode", "Error")
	network.plot(accuracy, "mediumvioletred", "accuracy_over_each_epoch_mnist_classification", "Episode", "Learning accuracy")
	network.test(model, tester, output)