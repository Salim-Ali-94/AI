import neural_network as network


if __name__ == "__main__":

	X, y = network.extract("mnist", "MNIST")
	Y = [element.item() for element in y]
	output, window = len(list(set(Y))), 3
	batch, train_percent = 32, 80
	episodes, learning_rate = 10, 5e-3
	trainer, tester, validater = network.partition(X, y, output, batch, train_percent)
	height, channel = X[0].shape[-1], X[0].shape[0]
	kernel, stride, padding = [5, 5], [1, 1], [1, 1]
	convolutions = [channel, 10, 20]
	activity = ["relu", "relu"]
	activity += ["relu", ""]
	neurons = [50, output]
	cost = "crossentropy"
	optimizer = "adam"
	model, error, accuracy = network.learn(trainer, neurons, activity, learning_rate, episodes, cost, optimizer, kernel, stride, padding, height, window, convolutions)
	network.plot(error, "forestgreen", "accumulated_error_over_each_epoch_mnist_classification_cnn", "Episode", "Error")
	network.plot(accuracy, "mediumvioletred", "accuracy_over_each_epoch_mnist_classification_cnn", "Episode", "Learning accuracy")
	network.test(model, tester, output, True)
