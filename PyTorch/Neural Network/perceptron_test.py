import neural_network as network


if __name__ == "__main__":

	batch, output = 50, 1
	train_percent = 80
	episodes = int(1e3)
	learning_rate = 1e-1
	encoding = [("b", 0), ("g", 1)]
	X, y = network.extract("ionosphere.txt", encoding = encoding, output = output)
	trainer, tester, validater = network.partition(X, y, output, batch, train_percent)
	dimension = len(X[0])
	neurons = (dimension, output)
	activity = ("sigmoid", )
	cost = "mse"
	optimizer = "adam"
	model, error, accuracy = network.learn(trainer, neurons, activity, learning_rate, episodes, optimizer, cost)
	network.plot(error, "forestgreen", "accumulated_error_over_each_epoch_binary_classification", "Episode", "Error")
	network.plot(accuracy, "mediumvioletred", "accuracy_over_each_epoch_binary_classification", "Episode", "Learning accuracy")
	network.test(model, tester, output)
