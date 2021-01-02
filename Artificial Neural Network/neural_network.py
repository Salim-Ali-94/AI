import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"


class Artificial_Neural_Network(object):

	normalize = lambda self, data: (data - data.min()) / (data.max() - data.min())
	activation = lambda self, weights, bias, activity, function, derivative: function(activity.dot(weights) + bias, derivative)

	def __init__(self, features, labels, hyper_parameters, learning_rate = 0.1, epochs = 100):

		self.hyper_parameters = hyper_parameters
		self.learning_rate = learning_rate
		self.features = self.normalize(features)
		self.layers = len(self.hyper_parameters)
		self.labels = labels
		self.epochs = epochs
		self.weights, self.biases = [], []

		for index in range(self.layers - 1):

			if (index == self.layers - 2):
				if (self.hyper_parameters[-1] == 1):
					bias = np.random.uniform(-1, 1)
				else:
					bias = np.random.uniform(-1, 1, self.hyper_parameters[index + 1])
			else:
				bias = np.random.uniform(-1, 1, self.hyper_parameters[index + 1])
			if (self.hyper_parameters[index] == 1):
				weight = np.random.uniform(-1, 1, self.hyper_parameters[index + 1])
			else:
				weight = np.random.uniform(-1, 1, [self.hyper_parameters[index], self.hyper_parameters[index + 1]])

			self.weights.append(weight)
			self.biases.append(bias)

		self.indicator = 0
		self.cost = []


	def sigmoid(self, x, derivative = False):

		if (derivative == False):
			return 1 / (1 + np.exp(-x.astype(float)))
		elif (derivative == True):
			return -np.exp(-x.astype(float)) / (1 + np.exp(-x.astype(float)))**2


	def hyperbolic_tangent(self, x, derivative = False):

		if (derivative == False):
			return np.tanh(x.astype(float))
		elif (derivative == True):
			return 1 / np.cosh(x.astype(float))**2


	def partition(self, training_percentage, validation_percentage = 0):

		train_factor = training_percentage / 100
		validate_factor = validation_percentage / 100
		total = len(self.labels)
		unique_labels = list(set(*self.labels.T))
		sections = len(unique_labels)
		portion = total // sections
		train_fraction = int(portion*train_factor)
		validate_fraction = int(portion*validate_factor)

		for index in range(sections):

			divide = int(portion*index)
			train_remainder = int(divide + train_fraction)
			validate_remainder = int(train_remainder + validate_fraction)
			step = portion*(index + 1)

			if (index == 0):
				if (validation_percentage == 0):
					trainingSet_label_previous = self.labels[divide:train_remainder]
					testSet_label_previous = self.labels[train_remainder:step]
					trainingSet_characteristic_previous = self.features[divide:train_remainder, :]
					testSet_characteristic_previous = self.features[train_remainder:step, :]
				else:
					trainingSet_label_previous = self.labels[divide:train_remainder]
					validationSet_label_previous = self.labels[train_remainder:validate_remainder]
					testSet_label_previous = self.labels[validate_remainder:step]
					trainingSet_characteristic_previous = self.features[divide:train_remainder, :]
					validationSet_characteristic_previous = self.features[train_remainder:validate_remainder, :]
					testSet_characteristic_previous = self.features[validate_remainder:step, :]
			else:
				if (validation_percentage == 0):
					trainingSet_label = self.labels[divide:train_remainder]
					testSet_label = self.labels[train_remainder:step]
					trainingSet_characteristic = self.features[divide:train_remainder, :]
					testSet_characteristic = self.features[train_remainder:step, :]
					trainingSet_outputs = np.concatenate((trainingSet_label_previous, trainingSet_label))
					testSet_outputs = np.concatenate((testSet_label_previous, testSet_label))
					trainingSet_inputs = np.concatenate((trainingSet_characteristic_previous, trainingSet_characteristic))
					testSet_inputs = np.concatenate((testSet_characteristic_previous, testSet_characteristic))
					trainingSet_label_previous = np.copy(trainingSet_outputs)
					testSet_label_previous = np.copy(testSet_outputs)
					trainingSet_characteristic_previous = np.copy(trainingSet_inputs)
					testSet_characteristic_previous = np.copy(testSet_inputs)
				else:
					trainingSet_label = self.labels[divide:train_remainder]
					validationSet_label = self.labels[train_remainder:validate_remainder]
					testSet_label = self.labels[validate_remainder:step]
					trainingSet_characteristic = self.features[divide:train_remainder, :]
					validationSet_characteristic = self.features[train_remainder:validate_remainder, :]
					testSet_characteristic = self.features[validate_remainder:step, :]
					trainingSet_outputs = np.concatenate((trainingSet_label_previous, trainingSet_label))
					validationSet_outputs = np.concatenate((validationSet_label_previous, validationSet_label))
					testSet_outputs = np.concatenate((testSet_label_previous, testSet_label))
					trainingSet_inputs = np.concatenate((trainingSet_characteristic_previous, trainingSet_characteristic))
					validationSet_inputs = np.concatenate((validationSet_characteristic_previous, validationSet_characteristic))
					testSet_inputs = np.concatenate((testSet_characteristic_previous, testSet_characteristic))
					trainingSet_label_previous = np.copy(trainingSet_outputs)
					validationSet_label_previous = np.copy(validationSet_outputs)
					testSet_label_previous = np.copy(testSet_outputs)
					trainingSet_characteristic_previous = np.copy(trainingSet_inputs)
					validationSet_characteristic_previous = np.copy(validationSet_inputs)
					testSet_characteristic_previous = np.copy(testSet_inputs)
					self.validation_inputs = np.copy(validationSet_inputs)
					self.validation_outputs = np.copy(validationSet_outputs)

		self.features = np.copy(trainingSet_inputs)
		self.labels = np.copy(trainingSet_outputs)
		self.test_inputs = np.copy(testSet_inputs)
		self.test_outputs = np.copy(testSet_outputs)
		self.indicator = 1


	def feed_forward(self, sample):

		Activity, Derivative = [], []

		for layer in range(self.layers - 1):

			if (layer == 0):
				activity = self.activation(self.weights[layer], self.biases[layer], self.features[sample], self.hyperbolic_tangent, False)
				delta_activity = self.activation(self.weights[layer], self.biases[layer], self.features[sample], self.hyperbolic_tangent, True)
			else:
				activity = self.activation(self.weights[layer], self.biases[layer], previous_activity, self.hyperbolic_tangent, False)
				delta_activity = self.activation(self.weights[layer], self.biases[layer], previous_activity, self.hyperbolic_tangent, True)

			previous_activity = np.copy(activity)
			Derivative.append(delta_activity)
			Activity.append(activity)

		return Activity, Derivative


	def back_propagation(self, sample):

		Delta = []
		activity, derivative = self.feed_forward(sample)

		for layer in range(self.layers - 2, -1, -1):

			if (layer == self.layers - 2):
				error = activity[layer] - self.labels[sample]
				delta = error*derivative[layer]
			else:
				if (len(delta) > 1):
					delta = delta.dot(self.weights[layer + 1].T)*derivative[layer]
				else:
					delta = delta*self.weights[layer + 1].T*derivative[layer]
			if (delta.shape[0] == 1):
				if (isinstance(delta[0], np.ndarray)):
					delta = np.copy(delta[0])

			Delta.append(delta)

		Delta = list(reversed(Delta))

		return Delta, activity


	def train(self):

		error = 0

		for episode in range(self.epochs):

			for example in range(self.features.shape[0]):

				delta, activity = self.back_propagation(example)

				for layer in range(self.layers - 2, -1, -1):

					if (layer == self.layers - 2):
						update = -delta[layer][np.newaxis].T*activity[layer - 1]
						update = np.copy(update.T)
					elif (layer == 0):
						update = -self.features[example][np.newaxis].T*delta[layer]
					elif (layer > 0):
						update = -activity[layer - 1][np.newaxis].T*delta[layer]

					self.weights[layer] = self.weights[layer] + self.learning_rate*update
					self.biases[layer] = self.biases[layer] + self.learning_rate*delta[layer]

				if (len(activity[-1]) > 1):
					error += (sum(activity[-1] - self.labels[example]))**2 / 2
				else:
					error += ((activity[-1] - self.labels[example])[0])**2 / 2

			print("Episode:", episode + 1) 
			print("Error:", error, "\n\n")
			error = np.array([error])
			self.cost.append(error)
			error = 0


	def classifier(self, data):

		for layer in range(self.layers - 1):

			if (layer == 0):
				activity = self.activation(self.weights[layer], self.biases[layer], data, self.hyperbolic_tangent, False)
			else:
				activity = self.activation(self.weights[layer], self.biases[layer], activity, self.hyperbolic_tangent, False)

		return activity


	def tester(self, data = None, target = None):

		miss_classification = 0
		sucessful_classification = 0

		if (self.indicator == 0):

			for index in range(data.shape[0]):

				output = np.round(self.classifier(data[index]), decimals = 1)

				if (output == target[index]):
					sucessful_classification += 1
				elif (output != target[index]):
					miss_classification += 1

		elif (self.indicator != 0):

			for index in range(self.test_inputs.shape[0]):

				output = np.round(self.classifier(self.test_inputs[index]), decimals = 1)

				if (output == self.test_outputs[index]):
					sucessful_classification += 1
				elif (output != self.test_outputs[index]):
					miss_classification += 1

		if (miss_classification == 1):
			print("The classifier correctly labeled {} input samples "\
			      "and incorrectly labeled {} sample from the test "\
			      "dataset\n\n".format(sucessful_classification, miss_classification))
		else:
			print("The classifier correctly labeled {} input samples "\
			      "and incorrectly labeled {} samples from the test "\
			      "dataset\n\n".format(sucessful_classification, miss_classification))


	def plotter(self):

		episodes = range(1, self.epochs + 1)
		plt.figure()
		axis = plt.axes(facecolor = "#E6E6E6")
		axis.set_axisbelow(True)
		plt.grid(color = "w", linestyle = "solid")

		for spine in axis.spines.values():
			spine.set_visible(False)

		plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
		plt.tick_params(axis = "y", which = "both", left = False, right = False)
		plt.plot(episodes, self.cost, color = "blue", linewidth = 1)
		plt.xlabel('Episode')
		plt.ylabel('Total square error per episode')
		plt.savefig('accumulated_square_errors_over_each_epoch.png', bbox_inches = "tight", dpi = 200)    
		plt.show()
