import neural_network as network


if __name__ == "__main__":

	flag, batch = None, 10
	train_percent, validate_percent = 80, 0
	episodes = int(1e3)
	learning_rate = 5e-3
	output = 3
	encoding = [("setosa", 0), ("versicolor", 1), ("virginica", 2)]
	X, y = network.extract("iris.csv", encoding, output, "species")
	dimension = len(X[0])
	neurons = (dimension, 10, 20, 20, 5, output)
	trainer, tester, validater = network.partition(X, y, output, batch, train_percent)
	activity = ("relu", "relu", "relu", "relu", "")
	cost = "crossentropy"
	optimizer = "adam"
	model, error, accuracy = network.learn(trainer, neurons, activity, learning_rate, episodes, optimizer, cost)
	network.plot(error, "forestgreen", "accumulated_errors_over_each_epoch_multilabel_classification", "Episode", "Error")
	network.plot(accuracy, "mediumvioletred", "accuracy_over_each_epoch_multilabel_classification", "Episode", "Learning accuracy")
	network.test(model, tester, output)
