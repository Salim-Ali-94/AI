import neural_network as network


if __name__ == "__main__":

	batch, output = 10, 3
	train_percent = 80
	episodes = int(1e3)
	learning_rate = 1e-2
	encoding = [("setosa", 0), ("versicolor", 1), ("virginica", 2)]
	X, y = network.extract("iris.csv", encoding = encoding, output = output, label = "species")
	dimension = len(X[0])
	neurons = [dimension, 10, 20, 20, 5, output]
	trainer, tester, validater = network.partition(X, y, output, batch, train_percent)
	activity = ["relu", "relu", "relu", "relu", ""]
	cost = "crossentropy"
	optimizer = "adam"
	model, error, accuracy = network.learn(trainer, neurons, activity, learning_rate, episodes, cost, optimizer)
	network.plot(error, "forestgreen", "accumulated_errors_over_each_epoch_multilabel_classification", "Episode", "Error")
	network.plot(accuracy, "mediumvioletred", "accuracy_over_each_epoch_multilabel_classification", "Episode", "Learning accuracy")
	network.test(model, tester, output)
